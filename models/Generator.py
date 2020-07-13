import torch
import torch.nn as nn

class Generator(nn.Module):
    #mnist out:28x28
    def __init__(self, z_dim=20):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),#out:N, 64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),#out:N, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, z):
        x = self.layer1(z)
        x = self.layer2(x)
        x = x.view(z.shape[0], 128, 7, 7)
        x = self.layer3(x)
        out = self.last(x)

        return out
