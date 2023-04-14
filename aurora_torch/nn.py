import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(
            # shape = (N, 1, 85, 128)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # shape = (N, 32, 85, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # shape = (N, 32, 42, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # shape = (N, 64, 42, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # shape = (N, 64, 21, 32)
            nn.Flatten(),
            # shape = (N, 64*21*32)
            nn.Linear(in_features=64*21*32, out_features=4),
        )


    def forward(self, x):
        x = self.net(x)
        return x