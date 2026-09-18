"""
Unit and Integration Tests for Diabetic Retinopathy Diagnostic Pipeline.
Verifies preprocessing, ResNet50 model inference, Grad-CAM attribution, and patch sensitivity.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from src.preprocessing import (
    crop_retina_border,
    apply_circular_mask,
    ben_graham_preprocessing,
    preprocess_for_inference
)
from src.model import build_model, predict_fundus, DR_CLASSES
from src.xai import GradCAM, overlay_heatmap_on_image, compute_patch_attribution


def test_preprocessing():
    print("Testing Preprocessing Pipeline...")
    # Create synthetic test fundus
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_img[50:250, 50:250] = [180, 80, 40]
    
    # 1. Border crop
    cropped = crop_retina_border(dummy_img)
    assert cropped.shape[0] <= 300 and cropped.shape[1] <= 300
    print("  [OK] crop_retina_border passed")
    
    # 2. Ben Graham Preprocessing
    bg_img = ben_graham_preprocessing(dummy_img, target_size=224)
    assert bg_img.shape == (224, 224, 3)
    assert bg_img.dtype == np.uint8
    print("  [OK] ben_graham_preprocessing passed")
    
    # 3. Inference Preprocessor
    pil_img = Image.fromarray(dummy_img)
    preprocessed_np, tensor_batch = preprocess_for_inference(pil_img, target_size=224)
    assert preprocessed_np.shape == (224, 224, 3)
    assert tensor_batch.shape == (1, 3, 224, 224)
    assert isinstance(tensor_batch, torch.Tensor)
    print("  [OK] preprocess_for_inference passed")


def test_model_inference():
    print("\nTesting Model Inference...")
    ckpt_path = "models/dr_resnet50_baseline.pth"
    model = build_model(
        model_name="resnet50",
        num_classes=5,
        pretrained=False,
        checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device="cpu"
    )
    
    dummy_tensor = torch.randn(1, 3, 224, 224)
    results = predict_fundus(model, dummy_tensor, device="cpu")
    
    assert "pred_class" in results
    assert 0 <= results["pred_class"] <= 4
    assert "confidence" in results
    assert 0.0 <= results["confidence"] <= 1.0
    assert "probabilities" in results
    assert len(results["probabilities"]) == 5
    assert "triage" in results
    assert "recommendation" in results
    print(f"  [OK] Model forward pass passed (Predicted: {results['short_name']}, Conf: {results['confidence']*100:.1f}%)")
    return model


def test_explainability(model):
    print("\nTesting Explainable AI (Grad-CAM & Patch Attribution)...")
    cam = GradCAM(model)
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    # 1. Grad-CAM
    heatmap, target_cls, score = cam.generate_heatmap(dummy_tensor, target_class=2)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) <= 1.0
    assert 0.0 <= np.max(heatmap) <= 1.0
    print(f"  [OK] Grad-CAM heatmap generated (Target Class: {target_cls}, Score: {score:.3f})")
    
    # 2. Overlay
    dummy_rgb = np.full((224, 224, 3), 120, dtype=np.uint8)
    overlay = overlay_heatmap_on_image(dummy_rgb, heatmap, alpha=0.5, threshold=0.2)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
    print("  [OK] Heatmap overlay blending passed")
    
    # 3. Patch Sensitivity
    patch_map = compute_patch_attribution(model, dummy_tensor, target_class=target_cls, patch_size=56, device="cpu")
    assert patch_map.shape == (224, 224)
    assert 0.0 <= np.max(patch_map) <= 1.0
    print("  [OK] Patch sensitivity attribution passed")
    
    cam.remove_hooks()


def test_sample_cases():
    print("\nTesting Built-in Sample Fundus Images...")
    samples = [
        ("samples/grade_0_normal.png", 0),
        ("samples/grade_1_mild.png", 1),
        ("samples/grade_2_moderate.png", 2),
        ("samples/grade_3_severe.png", 3),
        ("samples/grade_4_proliferative.png", 4)
    ]
    for spath, expected_grade in samples:
        assert os.path.exists(spath), f"Missing sample file: {spath}"
        img = Image.open(spath).convert("RGB")
        preprocessed_np, tensor = preprocess_for_inference(img, 224)
        assert preprocessed_np.shape == (224, 224, 3)
    print("  [OK] All 5 sample files exist and preprocess correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Retinal Diagnostic AI Test Suite")
    print("=" * 60)
    test_preprocessing()
    model = test_model_inference()
    test_explainability(model)
    test_sample_cases()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)
