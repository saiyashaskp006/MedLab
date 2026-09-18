"""
Explainable AI (XAI) Engine for Retinal Fundus Image Diagnosis.
Implements Grad-CAM (Gradient-weighted Class Activation Mapping),
lesion hotspot thresholding, and patch-based occlusion attribution.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM implementation for clinical decision explanation.
    Calculates gradient-weighted feature maps at the target convolutional layer
    to highlight retinal areas influencing the diagnostic decision.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Default to ResNet50 layer4 final block if target_layer not specified
        if target_layer is None:
            if hasattr(model, "get_cam_target_layer"):
                self.target_layer = model.get_cam_target_layer()
            elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                self.target_layer = model.backbone.layer4[-1]
            else:
                raise ValueError("Could not automatically identify target convolutional layer.")
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])
        
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates Grad-CAM heatmap for a single image tensor [1, 3, H, W].
        
        Args:
            input_tensor: Normalized PyTorch tensor with shape [1, 3, H, W].
            target_class: Target class index. If None, uses the highest predicted class.
            
        Returns:
            heatmap: Normalized float numpy array [0.0, 1.0] with shape (H, W).
            predicted_class: Class index used for attribution.
            score: Raw logit/softmax score for the target class.
        """
        self.model.zero_grad()
        
        # Ensure tensor requires gradient
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
            
        score = logits[0, target_class]
        
        # Backward pass to compute gradients with respect to target class
        score.backward(retain_graph=True)
        
        # Gradients shape: [1, C, H_feat, W_feat]
        gradients = self.gradients
        # Activations shape: [1, C, H_feat, W_feat]
        activations = self.activations
        
        # Global average pooling of gradients along spatial dimensions (alpha weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Linear combination of activations weighted by importance
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU to only keep features with a positive influence on target class
        cam = torch.relu(cam)
        
        # Squeeze to 2D numpy
        cam_np = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        # Resize to match input image resolution
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam_np, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        
        return heatmap_resized, target_class, float(score.item())
    
    def remove_hooks(self):
        """Clean up PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap onto RGB fundus image.
    
    Args:
        image_rgb: RGB image numpy array (H, W, 3) in range [0, 255].
        heatmap: 2D float heatmap in range [0.0, 1.0].
        alpha: Blending weight for heatmap (0.0 = original image, 1.0 = heatmap only).
        colormap: OpenCV colormap (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO).
        threshold: Minimum activation cutoff (0.0 - 1.0) to filter background noise.
        
    Returns:
        blended_rgb: Blended RGB image (H, W, 3) uint8.
    """
    # Resize heatmap to match image dimensions if needed
    h, w = image_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        
    # Apply optional threshold
    if threshold > 0.0:
        heatmap = np.where(heatmap >= threshold, heatmap, 0.0)
        # Re-normalize above threshold if non-empty
        active_vals = heatmap[heatmap > 0]
        if len(active_vals) > 0 and np.max(active_vals) > 0:
            heatmap = heatmap / np.max(active_vals)
            
    # Convert heatmap to uint8 colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Alpha blend with original fundus image
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
    return blended


def compute_patch_attribution(
    model: nn.Module,
    tensor_batch: torch.Tensor,
    target_class: int,
    patch_size: int = 32,
    device: str = "cpu"
) -> np.ndarray:
    """
    Occlusion Sensitivity / LIME-style Patch Importance Analysis:
    Divides image into a grid of patches, occludes each patch with the local average,
    and measures the drop in target class probability.
    
    High drop in probability indicates that the occluded retinal patch was critical
    for the model's diagnostic conclusion.
    """
    model.eval()
    tensor_batch = tensor_batch.to(device)
    
    with torch.no_grad():
        base_logits = model(tensor_batch)
        base_prob = torch.softmax(base_logits, dim=1)[0, target_class].item()
        
    _, _, h, w = tensor_batch.shape
    importance_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            # Create occluded copy
            perturbed = tensor_batch.clone()
            # Occlude patch with zero/neutral mean
            perturbed[0, :, y:y_end, x:x_end] = 0.0
            
            with torch.no_grad():
                pert_logits = model(perturbed)
                pert_prob = torch.softmax(pert_logits, dim=1)[0, target_class].item()
                
            # Drop in probability is attribution strength
            importance = max(0.0, base_prob - pert_prob)
            importance_map[y:y_end, x:x_end] = importance
            
    # Normalize importance map
    max_val = np.max(importance_map)
    if max_val > 1e-6:
        importance_map = importance_map / max_val
        
    return importance_map
