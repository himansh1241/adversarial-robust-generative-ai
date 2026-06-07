import torch

def fgsm_attack(model, image, label, epsilon=0.1):
    """
    Fast Gradient Sign Method (FGSM):
    Adds a tiny amount of noise in the direction that increases the loss.
    This tricks the model into making wrong predictions.
    
    epsilon: strength of the attack (higher = more visible noise, stronger attack)
    """
    image = image.clone().detach().requires_grad_(True)
    label = label.clone().detach()

    output = model(image)
    loss = torch.nn.functional.binary_cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    # Add perturbation in the sign of the gradient
    perturbation = epsilon * image.grad.sign()
    adversarial_image = image + perturbation

    # Clamp to valid pixel range [-1, 1]
    adversarial_image = torch.clamp(adversarial_image, -1, 1)
    return adversarial_image.detach()