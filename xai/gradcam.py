import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_saliency_map(model, image, label):
    """
    Saliency Map: Shows which pixels the model pays attention to.
    Computed as gradient of the output w.r.t. input pixels.
    """
    image = image.clone().detach().requires_grad_(True)
    label = label.clone().detach()

    output = model(image)
    loss = torch.nn.functional.binary_cross_entropy(output, label)
    loss.backward()

    saliency = image.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    return saliency.squeeze().numpy()

def plot_comparison(original, adversarial, defended, saliency_orig, saliency_adv):
    """
    Plots a side-by-side comparison of:
    original | adversarial | defended | saliency (original) | saliency (adversarial)
    """
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor('none')

    def to_numpy(img):
        arr = img.squeeze().detach().numpy()
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