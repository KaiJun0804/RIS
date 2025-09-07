import argparse
import glob
import os
from typing import List

import cv2
import torch

from src.models.frcnn_factory import create_faster_rcnn_model
from src.utils.common import is_image_file, increment_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Faster R-CNN license plate detector")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights (state_dict)")
    parser.add_argument("--source", type=str, required=True, help="Image file/dir/glob")
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-txt", action="store_true", help="Save detections to txt files")
    return parser.parse_args()


def load_images(source: str) -> List[str]:
    files: List[str] = []
    if os.path.isdir(source):
        for f in os.listdir(source):
            path = os.path.join(source, f)
            if os.path.isfile(path) and is_image_file(path):
                files.append(path)
    elif os.path.isfile(source):
        files = [source]
    else:
        files = [p for p in glob.glob(source, recursive=True) if is_image_file(p)]
    if not files:
        raise RuntimeError(f"No images found for source: {source}")
    return sorted(files)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = create_faster_rcnn_model(num_classes=2, use_pretrained=False)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    save_dir = increment_path(os.path.join("runs", "detect"), name="exp")
    os.makedirs(save_dir, exist_ok=True)
    if args.save_txt:
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    image_paths = load_images(args.source)
    for img_path in image_paths:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"Warning: failed to read {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        with torch.no_grad():
            outputs = model([image_tensor.to(device)])
        output = outputs[0]
        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()

        keep = scores >= args.conf_thres
        boxes = boxes[keep]
        scores = scores[keep]

        # draw and save
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, f"plate {score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, image_bgr)

        if args.save_txt:
            label_path = os.path.join(save_dir, "labels", os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            with open(label_path, "w") as f:
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = [float(v.item()) for v in box]
                    f.write(f"plate {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {score:.4f}\n")

    print(f"Saved results to {save_dir}")


if __name__ == "__main__":
    main()