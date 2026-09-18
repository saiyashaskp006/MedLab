"""
Medical Image Preprocessing for Diabetic Retinopathy Fundus Imaging.
Implements Ben Graham's illumination equalization, circular border cropping,
CLAHE contrast enhancement, and PyTorch transformation pipelines.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_retina_border(img_rgb: np.ndarray, tol: int = 10) -> np.ndarray:
    """
    Crops the non-informative black margin surrounding the circular retinal fundus.
    
    Args:
        img_rgb: Input RGB image as numpy array (H, W, 3).
        tol: Grayscale intensity tolerance for thresholding black borders.
        
    Returns:
        Cropped RGB numpy array containing only the fundus circle.
    """
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    
    # Check if image is completely dark
    if not np.any(mask):
        return img_rgb
    
    # Find bounding box of positive mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    cropped = img_rgb[ymin:ymax + 1, xmin:xmax + 1]
    return cropped


def apply_circular_mask(img: np.ndarray) -> np.ndarray:
    """
    Applies a smooth circular mask to remove any residual perimeter artifacts
    outside the retinal field of view.
    """
    h, w = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    radius = int(min(h, w) * 0.48)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img


def ben_graham_preprocessing(
    img_rgb: np.ndarray,
    target_size: int = 224,
    sigma_x: int = 10,
    apply_circle_mask: bool = True
) -> np.ndarray:
    """
    Industry-standard Ben Graham retinal preprocessing:
    1. Crops black background perimeter.
    2. Resizes to target dimension (e.g. 224x224).
    3. Normalizes uneven lighting via weighted Gaussian blur subtraction.
    4. Optionally applies a circular mask to eliminate border noise.
    
    Formula: 4 * img - 4 * GaussianBlur(img, sigma_x) + 128
    """
    # 1. Crop dark borders
    cropped = crop_retina_border(img_rgb, tol=12)
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        cropped = img_rgb
        
    # 2. Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # 3. Ben Graham's lighting equalization
    # Subtracting local average color normalizes variations in pupil dilation and camera lighting
    blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma_x)
    equalized = cv2.addWeighted(resized, 4, blurred, -4, 128)
    
    # 4. Optional circular mask
    if apply_circle_mask:
        equalized = apply_circular_mask(equalized)
        
    return equalized


def apply_clahe_green_channel(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the green channel (where retinal lesions and hemorrhages have highest optical contrast).
    """
    img_copy = img_rgb.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    img_copy[:, :, 1] = clahe.apply(img_copy[:, :, 1])
    return img_copy


def get_inference_transform(target_size: int = 224):
    """
    Returns standard torchvision transform pipeline for model inference.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_train_transform(target_size: int = 224):
    """
    Returns data augmentation pipeline for model training.
    Includes clinical invariance transforms: horizontal/vertical flips, affine rotations, and color jitter.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_for_inference(pil_or_np_image, target_size: int = 224):
    """
    End-to-end inference preprocessor:
    Accepts PIL Image or RGB numpy array, returns:
      1. preprocessed_np: Ben Graham preprocessed image for display (uint8 RGB, 0-255)
      2. tensor_batch: PyTorch tensor batch (1, 3, target_size, target_size) ready for model input.
    """
    if isinstance(pil_or_np_image, Image.Image):
        img_np = np.array(pil_or_np_image.convert('RGB'))
    else:
        img_np = pil_or_np_image
        
    preprocessed_np = ben_graham_preprocessing(img_np, target_size=target_size)
    
    # Convert preprocessed numpy to PIL then apply tensor transform
    pil_preprocessed = Image.fromarray(preprocessed_np)
    transform = get_inference_transform(target_size)
    tensor = transform(pil_preprocessed)
    tensor_batch = tensor.unsqueeze(0)  # Shape: [1, 3, H, W]
    
    return preprocessed_np, tensor_batch
