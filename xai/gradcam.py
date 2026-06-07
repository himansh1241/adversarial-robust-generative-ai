import torch
import matplotlib.pyplot as plt
from model.train import DEVICE

def compute_saliency_map(model, image, label):
    image = image.squeeze()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    image = image.to(DEVICE).clone().detach().requires_grad_(True)
    label = label.to(DEVICE).clone().detach()

    output = model(image)
    loss = torch.nn.functional.binary_cross_entropy(output, label)
    loss.backward()

    saliency = image.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    return saliency.squeeze().cpu().numpy()

def plot_comparison(original, adversarial, defended, saliency_orig, saliency_adv):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor('none')

    def to_numpy(img):
        arr = img.squeeze().detach().cpu().numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    titles = ["Original", "Adversarial", "Defended", "Saliency (orig)", "Saliency (adv)"]
    images = [
        to_numpy(original),
        to_numpy(adversarial),
        to_numpy(defended),
        saliency_orig,
        saliency_adv
    ]
    cmaps = ["gray", "gray", "gray", "hot", "hot"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig