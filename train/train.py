import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import CNNModel
from utils.dataset_loader import get_dataloaders
from utils.config import config

def train():
    # 1. 데이터 로딩
    data_path = config["data_path"]
    train_loader, val_loader, test_loader, class_map = get_dataloaders(data_path, batch_size=config["batch_size"])

    print("클래스 인덱스:", class_map)
    print("Train 배치 개수:", len(train_loader))
    print("Validation 배치 개수:", len(val_loader))
    print("Test 배치 개수:", len(test_loader))

    # 2. 모델 정의
    model = CNNModel(num_classes=len(class_map))

    # 3. 손실함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 4. 학습 루프
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        
        # 5. Validation 정확도 계산
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # (1) Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

        # (2) 항상 마지막 모델도 저장
        torch.save(model.state_dict(), "last_model.pth")

if __name__ == "__main__":
    train()
