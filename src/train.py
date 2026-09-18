"""
Training and Evaluation Pipeline for Diabetic Retinopathy Detection.
Features Stratified K-Fold splitting, Ben Graham retinal preprocessing,
Mixed Precision Training (AMP), Quadratic Weighted Kappa (QWK), and checkpointing.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.preprocessing import ben_graham_preprocessing, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model


class RetinalFundusDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy Fundus Images.
    Applies Ben Graham's illumination equalization and data augmentations.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        img_size: int = 224,
        is_training: bool = True,
        img_col: str = "id_code",
        label_col: str = "diagnosis"
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_training = is_training
        self.img_col = img_col
        self.label_col = label_col
        
        # Transforms after Ben Graham preprocessing
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        # Handle file extensions (.png or .jpeg or .jpg)
        if not (img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg')):
            img_path = os.path.join(self.img_dir, f"{img_name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, f"{img_name}.jpeg")
        else:
            img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Load RGB image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Apply Ben Graham preprocessing
        preprocessed_np = ben_graham_preprocessing(img_np, target_size=self.img_size)

        # Apply PyTorch transform
        tensor = self.transform(preprocessed_np)
        label = int(row[self.label_col]) if self.label_col in row else 0

        return tensor, label


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    use_cuda = device.type == "cuda"

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates model performance on validation/test set including Quadratic Weighted Kappa."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    total_samples = len(dataloader.dataset)
    val_loss = running_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    # Quadratic Weighted Kappa (standard clinical competition metric)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])

    return {
        "val_loss": val_loss,
        "accuracy": acc,
        "qwk": qwk,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": np.array(all_probs)
    }


def train_model(
    train_df: pd.DataFrame,
    img_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    val_size: float = 0.2,
    img_size: int = 224,
    save_path: str = "models/dr_resnet50_best.pth",
    device: Optional[str] = None
):
    """
    End-to-end training routine with stratified splitting and checkpointing.
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        
    print(f"Executing training on device: {device_obj}")

    # Stratified Train/Val split to maintain proportional DR severity distribution
    train_data, val_data = train_test_split(
        train_df,
        test_size=val_size,
        random_state=42,
        stratify=train_df["diagnosis"]
    )
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # DataLoaders
    train_dataset = RetinalFundusDataset(train_data, img_dir, img_size=img_size, is_training=True)
    val_dataset = RetinalFundusDataset(val_data, img_dir, img_size=img_size, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model & Optimization
    model = build_model(model_name="resnet50", num_classes=5, pretrained=True, device=str(device_obj))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device_obj.type == "cuda" else None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_qwk = -1.0
    history = []

    print("\n" + "="*60)
    print("Starting Training: Diabetic Retinopathy ResNet50")
    print("="*60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device_obj)
        val_metrics = evaluate(model, val_loader, criterion, device_obj)
        scheduler.step()
        elapsed = time.time() - start_time

        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["accuracy"]
        val_qwk = val_metrics["qwk"]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | QWK: {val_qwk:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_qwk": val_qwk
        })

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "val_loss": val_loss,
                "history": history
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (QWK: {best_qwk:.4f})")

    print(f"\nTraining Complete! Best Validation QWK: {best_qwk:.4f}")
    return model, history
