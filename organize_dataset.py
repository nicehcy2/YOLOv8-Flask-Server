"""
YOLO 라벨 기준으로 바운딩 박스 크롭 후 pothole / not_pothole 폴더로 분류

raw_data/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  data.yaml
"""

from pathlib import Path
from PIL import Image

RAW_DATA_DIR = "./raw_data"
OUTPUT_DIR = "./dataset"

# data.yaml 기준: 0=Drain Hole, 1=Pothole, 2=Sewer Cover
POTHOLE_CLASS_ID = 1

SPLITS = [("train", "train"), ("valid", "val")]


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # 이미지 범위 초과 방지
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


for src_split, dst_split in SPLITS:
    img_dir = Path(RAW_DATA_DIR) / src_split / "images"
    label_dir = Path(RAW_DATA_DIR) / src_split / "labels"

    pothole_out = Path(OUTPUT_DIR) / dst_split / "pothole"
    not_pothole_out = Path(OUTPUT_DIR) / dst_split / "not_pothole"
    pothole_out.mkdir(parents=True, exist_ok=True)
    not_pothole_out.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"[{src_split}] {img_dir} 없음, 스킵")
        continue

    crop_count = 0
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = img.crop((x1, y1, x2, y2))
            dest = pothole_out if class_id == POTHOLE_CLASS_ID else not_pothole_out
            save_name = f"{img_path.stem}_{i}.jpg"
            crop.save(dest / save_name)
            crop_count += 1

    print(f"[{dst_split}] pothole={len(list(pothole_out.iterdir()))}, not_pothole={len(list(not_pothole_out.iterdir()))} (총 크롭 {crop_count}개)")

print("완료")
