from typing import List, Tuple
import os
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in IMAGE_EXTENSIONS


def collate_fn(batch: List[Tuple]):
    return tuple(zip(*batch))


def increment_path(base_dir: str, name: str = "exp") -> str:
    """Return next available path like runs/detect/exp, exp1, exp2, ..."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pattern = re.compile(rf"^{re.escape(name)}(\d+)?$")
    indices = [int(m.group(1)) for d in existing for m in [pattern.match(d)] if m and m.group(1)]
    if f"{name}" not in existing:
        return os.path.join(base_dir, name)
    idx = 1 if not indices else max(indices) + 1
    return os.path.join(base_dir, f"{name}{idx}")