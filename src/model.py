"""
Model Architecture and Clinical Knowledge Base for Diabetic Retinopathy Diagnosis.
Provides transfer learning models (ResNet50 / InceptionV3) and clinical triage mapping.
"""

from typing import Dict, Any, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale
DR_CLASSES: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "No Diabetic Retinopathy",
        "short": "No DR",
        "severity": "Normal",
        "color": "#28a745",  # Green
        "findings": "No microvascular abnormalities detected. Normal optic disc, macula, and retinal vasculature.",
        "triage": "Routine Screening",
        "urgency": "Low (Annual follow-up)",
        "recommendation": "Maintain optimal glycemic control (HbA1c < 7.0%) and annual dilated eye screening."
    },
    1: {
        "name": "Mild Non-Proliferative DR",
        "short": "Mild NPDR",
        "severity": "Mild",
        "color": "#17a2b8",  # Teal / Blue
        "findings": "Presence of microaneurysms only (isolated focal capillary dilatations).",
        "triage": "Non-Urgent Clinical Review",
        "urgency": "Low to Moderate (6-12 months)",
        "recommendation": "Repeat retinal examination within 6-12 months. Review systemic diabetes and blood pressure control."
    },
    2: {
        "name": "Moderate Non-Proliferative DR",
        "short": "Moderate NPDR",
        "severity": "Moderate",
        "color": "#ffc107",  # Yellow / Amber
        "findings": "More than microaneurysms but less than Severe NPDR. Dot & blot hemorrhages, hard exudates (lipid leakage), or cotton-wool spots.",
        "triage": "Comprehensive Ophthalmology Referral",
        "urgency": "Moderate (Refer within 2-4 months)",
        "recommendation": "Schedule formal ophthalmologic evaluation. Screen for diabetic macular edema (DME) via Optical Coherence Tomography (OCT)."
    },
    3: {
        "name": "Severe Non-Proliferative DR",
        "short": "Severe NPDR",
        "severity": "Severe",
        "color": "#fd7e14",  # Orange
        "findings": "Complies with the 4-2-1 international rule: >20 intraretinal hemorrhages in 4 quadrants, definite venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant.",
        "triage": "Urgent Retina Specialist Referral",
        "urgency": "High (Refer within 2-4 weeks)",
        "recommendation": "High risk of rapid progression to Proliferative DR (~50% within 1 year). Immediate retina specialist referral."
    },
    4: {
        "name": "Proliferative Diabetic Retinopathy",
        "short": "Proliferative DR",
        "severity": "Vision-Threatening (High)",
        "color": "#dc3545",  # Red
        "findings": "Active neovascularization (new abnormal fragile vessels on optic disc [NVD] or elsewhere [NVE]), preretinal/vitreous hemorrhage, or fibrovascular proliferation.",
        "triage": "Emergency Retinal Intervention",
        "urgency": "Critical (Refer within 24-48 hours)",
        "recommendation": "Immediate intervention required (Panretinal Photocoagulation [PRP] laser surgery and/or anti-VEGF intravitreal therapy) to prevent irreversible blindness."
    }
}


class RetinalResNet50(nn.Module):
    """
    ResNet50 architecture customized for 5-stage Diabetic Retinopathy Classification.
    Provides gradient capture hooks on the final residual block for Grad-CAM explainability.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.3):
        super(RetinalResNet50, self).__init__()
        
        # Load torchvision pretrained weights (weights=ResNet50_Weights.DEFAULT)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # In_features from the average pooling layer (2048 for ResNet50)
        in_features = self.backbone.fc.in_features
        
        # Replace classification head with clinical classification MLP
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_cam_target_layer(self) -> nn.Module:
        """
        Returns the target convolutional layer for Grad-CAM.
        For ResNet50, this is the final bottleneck block of layer4.
        """
        return self.backbone.layer4[-1]


def build_model(
    model_name: str = "resnet50",
    num_classes: int = 5,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Builds and initializes the Diabetic Retinopathy deep learning model.
    Loads checkpoint weights if provided.
    """
    if model_name.lower() == "resnet50":
        model = RetinalResNet50(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Supported: ['resnet50']")
        
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle state_dict wrapped in 'state_dict' or 'model_state_dict'
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {checkpoint_path}")
        
    model.to(device)
    model.eval()
    return model


def predict_fundus(
    model: nn.Module,
    tensor_batch: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Runs model inference on preprocessed fundus tensor batch (1, 3, H, W).
    Returns comprehensive diagnostic outputs including class probabilities,
    predicted ICDR grade, and clinical triage recommendations.
    """
    model.eval()
    tensor_batch = tensor_batch.to(device)
    
    with torch.no_grad():
        logits = model(tensor_batch)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])
    
    class_info = DR_CLASSES[pred_idx]
    
    return {
        "pred_class": pred_idx,
        "class_name": class_info["name"],
        "short_name": class_info["short"],
        "severity": class_info["severity"],
        "color": class_info["color"],
        "confidence": confidence,
        "probabilities": {DR_CLASSES[i]["short"]: float(prob) for i, prob in enumerate(probabilities)},
        "raw_probabilities": probabilities.tolist(),
        "findings": class_info["findings"],
        "triage": class_info["triage"],
        "urgency": class_info["urgency"],
        "recommendation": class_info["recommendation"]
    }
