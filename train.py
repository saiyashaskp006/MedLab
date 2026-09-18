import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from src.data import build_loaders
from src.model import build_model


def evaluate(model, loader, device, num_classes):
    model.eval()
    targets, predictions, scores = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            scores.extend(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.extend(logits.argmax(1).cpu().numpy())
            targets.extend(labels.numpy())
    metrics = {
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "report": classification_report(targets, predictions, output_dict=True, zero_division=0),
    }
    try:
        metrics["macro_auc_ovr"] = float(roc_auc_score(targets, scores, multi_class="ovr"))
    except ValueError:
        metrics["macro_auc_ovr"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a DR classifier with transfer learning.")
    parser.add_argument("--data-dir", required=True, help="ImageFolder root containing one folder per grade.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="artifacts/best_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, classes = build_loaders(args.data_dir, args.batch_size)
    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
        metrics = evaluate(model, val_loader, device, len(classes))
        print(f"epoch={epoch + 1}/{args.epochs} loss={running_loss / len(train_loader.dataset):.4f} macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({"model_state": model.state_dict(), "classes": classes, "metrics": metrics}, output)
            (output.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint to {output}")


if __name__ == "__main__":
    main()
