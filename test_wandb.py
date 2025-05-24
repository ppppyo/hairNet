import wandb

print("🟡 Trying to initialize wandb...")
wandb.init(project="hairnet", config={"learning_rate": 0.001, "epochs": 5})
print("🟢 wandb initialized!")

for epoch in range(5):
    loss = 1.0 / (epoch + 1)
    acc = epoch * 10
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": acc})
    print(f"Epoch {epoch+1} logged.")

print("✅ Finished logging test.")
