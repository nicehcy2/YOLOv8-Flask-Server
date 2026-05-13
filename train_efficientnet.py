import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

DATA_DIR = "./dataset"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4
IMG_SIZE = 380
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_loader = DataLoader(
    ImageFolder(f"{DATA_DIR}/train", transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    ImageFolder(f"{DATA_DIR}/val", transform=transform),
    batch_size=BATCH_SIZE, shuffle=False
)

model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} | train_acc={correct/total:.4f} | val_acc={val_correct/val_total:.4f}")

torch.save(model.state_dict(), "efficientnet_b4.pth")
print("모델 저장 완료: efficientnet_b4.pth")
