from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGE_SIZE = 224


def build_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_loaders(data_dir: str, batch_size: int = 32, val_fraction: float = 0.2, seed: int = 42):
    root = Path(data_dir)
    dataset = datasets.ImageFolder(root, transform=build_train_transform())
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = __import__("torch").Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    validation_dataset = datasets.ImageFolder(root, transform=build_inference_transform())
    val_set.dataset = validation_dataset
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        dataset.classes,
    )
