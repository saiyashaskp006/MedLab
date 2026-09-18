from io import BytesIO

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        return cam / (cam.max() + 1e-8)

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    heat = (cm.get_cmap("magma")(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = (0.55 * np.asarray(base) + 0.45 * heat).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)
