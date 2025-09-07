## Faster R-CNN License Plate Detector (YOLO-style runs/detect outputs)

This project trains a Faster R-CNN model (PyTorch/Torchvision) to detect vehicle license plates using YOLO-format labels and outputs detections to a YOLO-like `runs/detect/exp*` directory. Model weights are saved as a `.pt` file.

### 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Dataset format (YOLO)

- Images and labels organized like YOLO datasets:
  - `DATA_ROOT/images/train/*.jpg|png|...`
  - `DATA_ROOT/images/val/*.jpg|png|...`
  - `DATA_ROOT/labels/train/*.txt`
  - `DATA_ROOT/labels/val/*.txt`
- Each `.txt` file contains lines: `class x_center y_center width height` (normalized [0,1])
- This repository assumes a single class `plate` with class id `0`. Any other classes in labels are ignored.

### 3) Train

```bash
python train.py \
  --data-root /path/to/dataset \
  --epochs 20 \
  --batch-size 4 \
  --lr 0.005 \
  --num-workers 4 \
  --device cuda  # or cpu
```

- Weights will be saved to `weights/plate_frcnn.pt` by default.

### 4) Inference (YOLO-like runs/detect/exp*)

```bash
python infer.py \
  --weights weights/plate_frcnn.pt \
  --source /path/to/images_or_dir_or_glob \
  --conf-thres 0.5 \
  --device cuda  # or cpu
```

- Outputs are saved to `runs/detect/exp*` with images annotated and per-image `.txt` detections (format: `plate x_min y_min x_max y_max score`).

### 5) Notes

- This project focuses on plate detection (bounding boxes). It does not perform OCR. You can integrate an OCR model later to read characters from the detected plates.
- If downloading pretrained weights is blocked, pass `--no-pretrained` during training to initialize randomly.