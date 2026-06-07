import torch
import torch.nn as nn
import torch.optim as optim
from model.gan import Generator, Discriminator
from utils.data_loader import get_data_loaders

NOISE_DIM = 100
EPOCHS = 10  # increase to 50+ for better quality
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gan(progress_callback=None, epochs_override=None):
    global EPOCHS
    if epochs_override is not None:
        EPOCHS = epochs_override
    """
    Trains the GAN. progress_callback is a Streamlit progress bar (optional).
    Returns the trained generator and discriminator.
    """
    train_loader, _ = get_data_loaders()
    generator     = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    opt_g = optim.Adam(generator.parameters(),     lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []

    for epoch in range(EPOCHS):
        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # --- Train Discriminator ---
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_imgs = generator(noise).detach()

            d_loss_real = criterion(discriminator(real_imgs), real_labels)
            d_loss_fake = criterion(discriminator(fake_imgs), fake_labels)
            d_loss = d_loss_real + d_loss_fake

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_imgs = generator(noise)
            g_loss = criterion(discriminator(fake_imgs), real_labels)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        print(f"Epoch [{epoch+1}/{EPOCHS}] | G loss: {g_loss.item():.4f} | D loss: {d_loss.item():.4f}")

        if progress_callback:
            progress_callback((epoch + 1) / EPOCHS)

    torch.save(generator.state_dict(),     "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    return generator, discriminator, g_losses, d_losses