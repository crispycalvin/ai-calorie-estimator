# Fine-tune ResNet18 on Food-101
# Usage (CPU okay but slow):  python -m backend.app.ml.train_food101 --epochs 2
# Faster on GPU (Colab):      python -m backend.app.ml.train_food101 --device cuda --epochs 10
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

def get_transforms(img_size=224):
    # Standard ImageNet normalization (works for ResNet18)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(Path.home() / "data" / "food-101"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")  # "cuda" if available
    p.add_argument("--weights_out", type=str, default="backend/app/ml/weights/food101_resnet18.pt")
    p.add_argument("--classes_out", type=str, default="backend/app/ml/weights/food101_classes.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "mps" else "cpu")
    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

    train_tf, val_tf = get_transforms(args.img_size)

    # TorchVision has Food-101 ready to download/split
    train_ds = datasets.Food101(root=args.data_dir, split="train", transform=train_tf, download=True)
    val_ds   = datasets.Food101(root=args.data_dir, split="test",  transform=val_tf,   download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    # ResNet18 backbone with new classifier head
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p_ in model.layer1.parameters(): p_.requires_grad = True
    for p_ in model.layer2.parameters(): p_.requires_grad = True
    for p_ in model.layer3.parameters(): p_.requires_grad = True
    for p_ in model.layer4.parameters(): p_.requires_grad = True
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        return loss_sum / max(1,total), correct / max(1,total)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = evaluate()
        print(f"[val] loss={val_loss:.4f} acc={val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = { "model": model.state_dict(), "acc": best_acc }

    # Save best weights + class names
    if best_state is None:
        best_state = { "model": model.state_dict(), "acc": best_acc }
    torch.save(best_state, args.weights_out)
    with open(args.classes_out, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved: {args.weights_out} (acc={best_state['acc']*100:.2f}%), classes: {args.classes_out}")

if __name__ == "__main__":
    main()
