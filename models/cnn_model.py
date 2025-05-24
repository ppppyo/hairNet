import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),               # → 256 x 256
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),              # → 256 x 256
            nn.MaxPool2d(2),                                         # → 128 x 128

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),              # → 128 x 128
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),              # → 128 x 128
            nn.MaxPool2d(2),                                         # → 64 x 64

            nn.Flatten(),

            nn.Linear(64*64*64, 256),  # 262144 → 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)
