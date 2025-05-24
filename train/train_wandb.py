import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.cnn_model import CNNModel
from utils.dataset_loader import get_dataloaders
from utils.config import config

from tqdm import tqdm

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ✅ 1. wandb 초기화
    if config.get("use_wandb", False):
        print("wandb about to initialize")
        wandb.init(project="hairnet", config=config, reinit=True)
        wandb_config = wandb.config
        print("✅ wandb initialized!")
    else:
        print("❌ wandb disabled!")
        wandb_config = config  # wandb 없이도 동일하게 사용

    # 2. 데이터 로딩
    train_loader, val_loader, test_loader, class_map = get_dataloaders(
        wandb_config["data_path"],
        batch_size=wandb_config["batch_size"]
    )

    model = CNNModel(num_classes=wandb_config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=wandb_config["learning_rate"],
        weight_decay=wandb_config.get("weight_decay", 0.01)
    )

    scheduler = None
    if wandb_config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=wandb_config.get("scheduler_step", 10),
            gamma=wandb_config.get("scheduler_gamma", 0.5)
        )

    start_epoch = 0
    best_val_acc = 0.0

    if config.get("resume", False):
        checkpoint = torch.load(config.get("checkpoint_path", "checkpoint.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"🔁 이어서 학습 시작: epoch {start_epoch}부터")

    for epoch in range(start_epoch, wandb_config["epochs"]):
        model.train()
        total_loss = 0.0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}", dynamic_ncols=True)

        for batch_idx, (images, labels) in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}", batch=batch_idx)


        # ✅ validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # ✅ wandb 로깅
        if config.get("use_wandb", False):
            wandb.log({
                "epoch": epoch + 1,
                "loss": total_loss,
                "val_accuracy": val_acc
            })

        # ✅ 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")
            # if config.get("use_wandb", False):
            #     wandb.save("best_model.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }

        # torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")

        if epoch + 1 == wandb_config["epochs"]:
            torch.save(model.state_dict(), "last_model.pth")    

        if scheduler:
            scheduler.step()


if __name__ == "__main__":
    train()

    
    if config.get("use_wandb", False):
        wandb.finish()
