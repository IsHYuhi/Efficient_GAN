import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, z_dim=20):
        super(Discriminator, self).__init__()

        self.x_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.x_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.z_layer1 = nn.Linear(z_dim, 512)


        self.last1 = nn.Sequential(
            nn.Linear(3648, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.last2  = nn.Linear(1024, 1)


    def forward(self, x, z):
        x = self.x_layer1(x)
        x = self.x_layer2(x)

        z = z.view(z.shape[0], -1)
        z = self.z_layer1(z)

        x = x.view(-1, 64 * 7 * 7)
        out = torch.cat([x, z], dim=1)
        out = self.last1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)#to 2d

        out = self.last2(out)

        return out, feature


# import matplotlib.pyplot as plt
# from Generator import Generator
# D = Discriminator(z_dim=20)
# G = Generator(z_dim=20)
# G.train()
# input_z = torch.randn(2, 20)
# fake_images = G(input_z)
# d_out, _ = D(fake_images, input_z)
# print(nn.Sigmoid()(d_out))#to 0~1
