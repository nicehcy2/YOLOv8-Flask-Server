import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "./efficientnet_b4.pth"
CLASSES = ["not_pothole", "pothole"]
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    model = models.efficientnet_b4()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def predict(model, img_path: str):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0]
        pred = prob.argmax().item()
    return CLASSES[pred], prob[pred].item()

if __name__ == "__main__":
    detect_dir = Path("./detect_images")
    if not detect_dir.exists():
        print("detect_images 폴더가 없습니다.")
        sys.exit(1)

    image_paths = [p for p in detect_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_paths:
        print("detect_images 폴더에 이미지가 없습니다.")
        sys.exit(1)

    model = load_model()
    print(f"총 {len(image_paths)}장 분류 시작\n")

    for path in sorted(image_paths):
        label, confidence = predict(model, str(path))
        print(f"{path.name} → {label} ({confidence:.4f})")
