import torch
import torch.nn as nn

class Generator(nn.Module):
    """Takes random noise, outputs a fake image."""
    def __init__(self, noise_dim=100, img_channels=1, img_size=28):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_channels * img_size * img_size),
            nn.Tanh()  # output between -1 and 1
        )
        self.img_channels = img_channels
        self.img_size = img_size

    def forward(self, noise):
        img = self.model(noise)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    """Takes an image, outputs probability of being real."""
    def __init__(self, img_channels=1, img_size=28):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # output 0 (fake) or 1 (real)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)