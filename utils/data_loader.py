import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """
    Loads MNIST dataset as a proxy for medical/sensitive images.
    In a real project, replace this with your actual image dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader