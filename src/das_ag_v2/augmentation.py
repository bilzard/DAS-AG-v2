import torch


def gaussian_noise(image, sigma=1.0):
    noise = torch.randn_like(image)
    return image + sigma * noise


def color_shift(image, shift: float = 1.0):
    N, C, H, W = image.shape
    mu = torch.zeros((N, C, 1, 1), device=image.device).uniform_(
        -shift, shift
    )  # U[-1,1]
    sigma = torch.exp(
        torch.zeros((N, C, 1, 1), device=image.device).uniform_(-shift, shift)
    )  # exp(U[-1,1])

    return sigma * image + mu


def add_positional_rolling(image, max_shift=10):
    N, C, H, W = image.shape
    jittered_images = torch.zeros_like(image)

    for i in range(N):
        dx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        dy = torch.randint(-max_shift, max_shift + 1, (1,)).item()

        jittered_images[i] = torch.roll(image[i], shifts=(dy, dx), dims=(1, 2))  # type: ignore

    return jittered_images
