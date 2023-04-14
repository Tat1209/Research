import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        # GPUが使える場合は、GPU使用モードにする。
        self.classes = 4
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*21*21, 4)
        )
        # self.classfier = None
        

    def forward(self, x):
        x = self.net(x)
        # if self.classfier is None: self.classfier = nn.Linear(in_features=x.shape[-1], out_features=self.classes).to(self.device)
        # x = self.classfier(x)
        return x
    


# """
            # shape = (N, 3, 85, 85)
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            # # shape = (N, 32, 85, 85)
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # # shape = (N, 32, 42, 42)
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # # shape = (N, 64, 42, 42)
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # shape = (N, 64, 21, 21)
            # nn.Flatten(),
            # # shape = (N, 64 * 21 * 21)
            # nn.Linear(64 * 21 * 21, 4096),
            # # shape = (N, 4096)
            # nn.ReLU(),
            # nn.Linear(4096, 1000),
            # # shape = (N, 1000)
            # nn.ReLU(),
            # nn.Linear(1000, 4)
            # # shape = (N, 4)
            # nn.AdaptiveAvgPool2d(5),
            # # shape = (N, 64, 5, 5)
            # nn.Flatten(),
            # # shape = (N, 64 * 5 * 5)
            # nn.Linear(64 * 5 * 5, 4),
# """