import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from src.datasets.yolo_plate_dataset import YOLOPlateDataset
from src.models.frcnn_factory import create_faster_rcnn_model
from src.utils.common import collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for license plates (YOLO-format)")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root with images/ and labels/ subfolders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use COCO pretrained weights")
    parser.add_argument("--weights-out", type=str, default="weights/plate_frcnn.pt")
    return parser.parse_args()


def make_datasets(data_root: str) -> Tuple[YOLOPlateDataset, YOLOPlateDataset]:
    train_images = os.path.join(data_root, "images", "train")
    train_labels = os.path.join(data_root, "labels", "train")
    val_images = os.path.join(data_root, "images", "val")
    val_labels = os.path.join(data_root, "labels", "val")

    train_ds = YOLOPlateDataset(train_images, train_labels)
    val_ds = YOLOPlateDataset(val_images, val_labels)
    return train_ds, val_ds


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_iou = 0.0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                gt = target["boxes"].to(device)
                pred = output["boxes"]
                if gt.numel() == 0 or pred.numel() == 0:
                    continue
                ious = box_iou(gt, pred).max(dim=1).values
                total_iou += ious.mean().item()
                total += 1
    return total_iou / max(total, 1)


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    train_ds, val_ds = make_datasets(args.data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = create_faster_rcnn_model(num_classes=2, use_pretrained=not args.no_pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()

        val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.4f} - val_mIoU: {val_iou:.4f}")

    torch.save(model.state_dict(), args.weights_out)
    print(f"Saved weights to {args.weights_out}")


if __name__ == "__main__":
    main()