from dataclasses import dataclass

import yaml


@dataclass
class Prompt:
    positive_prompts: tuple[str, ...] = ()
    negative_prompts: tuple[str, ...] = ()


@dataclass
class Config:
    # general optimization params
    num_steps: int = 100
    batch_size: int = 8
    lr: float = 0.1
    betas: tuple[float, float] = (0.5, 0.99)
    eta_min_ratio: float = 0.01
    eval_steps: int = 1
    checkpoint_interval: int = 100

    # Architecture
    image_resolutions: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    max_shift: int = 128
    full_size: int = 386

    # CLIP
    lambda_clip: float = 1.0
    nonlinear_loss_scaling: bool = True

    # Aesthetic
    reverse_aesthetic: bool = False
    aesthetic_range: tuple[float, float] = (0.0, 1.0)
    aesthetic_schedule: str = "exponential_decay"
    aesthetic_decay_rate: float = 0.02

    # Reproducibility
    use_deterministic_algorithm: bool = False
    mode: str = "bilinear"
    seed: int = 0

    # Regularization
    lambda_tv: float = 1e-8
    lambda_l1: float = 0

    # Augmentation
    apply_augmentation: bool = True

    ## Gaussian noise
    noise_schedule: str = "exponential_warmup"
    noise_decay_rate: float = 0.03
    noise_std_range: tuple[float, float] = (0.2, 0.5)

    ## Color Shift
    color_shift_schedule: str = "exponential_warmup"
    color_shift_decay_rate: float = 0.03
    color_shift_range: tuple[float, float] = (0.05, 0.3)

    ## Positional Jitter
    pos_jitter_schedule: str = "exponential_warmup"
    pos_jitter_decay_rate: float = 0.03


def load_yaml_to_dataclass(yaml_file: str, dataclass_type):
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dataclass_type(**data)
