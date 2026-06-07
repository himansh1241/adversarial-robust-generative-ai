import torch

def pgd_attack(model, image, label, epsilon=0.1, alpha=0.01, num_steps=40):
    """
    Projected Gradient Descent (PGD) - a stronger, iterative version of FGSM.
    Takes many small steps instead of one big step.
    
    epsilon: maximum total perturbation
    alpha: step size per iteration
    num_steps: number of iterations
    """
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()

    for _ in range(num_steps):
        perturbed_image.requires_grad_(True)
        output = model(perturbed_image)
        loss = torch.nn.functional.binary_cross_entropy(output, label)

        model.zero_grad()
        loss.backward()

        # Take one small FGSM step
        step = alpha * perturbed_image.grad.sign()
        perturbed_image = perturbed_image.detach() + step

        # Project back into epsilon-ball around original image
        delta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(original_image + delta, -1, 1).detach()

    return perturbed_image