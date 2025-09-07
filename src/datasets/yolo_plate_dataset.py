from typing import Dict, List, Tuple
import os
from PIL import Image
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor

from src.utils.common import is_image_file


class YOLOPlateDataset(torch.utils.data.Dataset):
    """YOLO-format dataset for single-class 'plate' detection.

    Assumed structure:
    - images_dir: contains images
    - labels_dir: contains matching .txt files with YOLO annotations

    Label format per line: class_id x_center y_center width height (all normalized [0,1]).
    Only class_id==0 is used (plate). Others are ignored.
    """

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if is_image_file(os.path.join(images_dir, f))]
        )
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_labels(self, label_path: str, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        if not os.path.exists(label_path):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5 and len(parts) != 6:
                    # support optional confidence column on last position
                    continue
                cls_id = int(parts[0])
                if cls_id != 0:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                # Convert normalized xywh to xyxy in absolute pixels
                xc *= width
                yc *= height
                w *= width
                h *= height
                x1 = max(0.0, xc - w / 2.0)
                y1 = max(0.0, yc - h / 2.0)
                x2 = min(float(width - 1), xc + w / 2.0)
                y2 = min(float(height - 1), yc + h / 2.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # plate class index for Faster R-CNN (background=0, plate=1)
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, idx: int):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes_np, labels_np = self._read_labels(label_path, width, height)

        # Convert to tensors for Faster R-CNN
        image_tensor = to_tensor(image)
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        labels = torch.as_tensor(labels_np, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() else torch.tensor([], dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target: Dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target