import gc

import torch
import torch.nn as nn
import torch.nn.functional as TF

from das_ag_v2.augmentation import (
    add_positional_rolling,
    color_shift,
    gaussian_noise,
)
from das_ag_v2.config import Config
from das_ag_v2.loss import l1_loss, total_variation_loss
from das_ag_v2.model import AestheticModelV2, inv_process
from das_ag_v2.schedule import schedule_map


def interpolate(image, target_size, mode="bicubic"):
    return TF.interpolate(
        image,
        size=(target_size, target_size),
        mode=mode,
        **({"align_corners": False} if mode == "bicubic" or mode == "bilinear" else {}),
    )


class DasGenerator:
    def __init__(
        self, predictor_path: str, clip_model_path: str, config: Config, device="cuda"
    ):
        self.model = AestheticModelV2(predictor_path, clip_model_path).to(device)

        self.device = device
        self.images = None
        self.config = config
        self.mode = config.mode
        self.checkpoints = []
        self.scores = []
        self.noise_schedule = schedule_map[config.noise_schedule]
        self.color_shift_schedule = schedule_map[config.color_shift_schedule]
        self.pos_jitter_schedule = schedule_map[config.pos_jitter_schedule]
        self.aesthetic_schedule = schedule_map[config.aesthetic_schedule]
        self.max_size = self.config.image_resolutions[-1]
        self.full_size = self.config.full_size

        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def generate(
        self, pos_texts: list[str], neg_texts: list[str], progress_callback=None
    ):
        cfg = self.config
        image_stack = [
            (nn.Parameter(torch.randn(1, 3, s, s, device=self.device) / s))
            for s in cfg.image_resolutions
        ]

        optimizer = torch.optim.Adam(image_stack, lr=cfg.lr, betas=cfg.betas)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_steps, eta_min=cfg.eta_min_ratio * cfg.lr
        )
        self.model.text_encoder.eval()
        self.model.predictor.train()
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            noise_std = self.noise_schedule(
                step,
                cfg.num_steps,
                start=cfg.noise_std_range[0],
                end=cfg.noise_std_range[1],
                rate=cfg.noise_decay_rate,
            )
            c_shift = self.color_shift_schedule(
                step,
                cfg.num_steps,
                start=cfg.color_shift_range[0],
                end=cfg.color_shift_range[1],
                rate=cfg.color_shift_decay_rate,
            )
            pos_shift = int(
                self.pos_jitter_schedule(
                    step,
                    cfg.num_steps,
                    start=cfg.max_shift,
                    end=0,
                    rate=cfg.pos_jitter_decay_rate,
                )
            )
            lambda_aesthetic = self.aesthetic_schedule(
                step,
                cfg.num_steps,
                start=cfg.aesthetic_range[0],
                end=cfg.aesthetic_range[1],
                rate=cfg.aesthetic_decay_rate,
            )
            image = torch.stack(
                [interpolate(i, self.max_size, mode=self.mode) for i in image_stack]
            ).mean(0)
            image = image.tanh()
            if cfg.apply_augmentation:
                images = image.repeat(cfg.batch_size, 1, 1, 1)

                images = add_positional_rolling(images, max_shift=pos_shift)
                images = color_shift(images, c_shift)
                images = gaussian_noise(images, sigma=noise_std)
            else:
                images = image.repeat(1, 1, 1, 1)

            self.pad = pad = cfg.max_shift
            images = images[..., pad:-pad, pad:-pad]
            if images.shape[-1] != self.full_size:
                images = interpolate(images, self.full_size, mode=self.mode)

            aesthetic_score, pos_clip, neg_clip = self.model.calc_scores(
                images, pos_texts, neg_texts
            )
            if cfg.nonlinear_loss_scaling:
                pos_clip = pos_clip.arctanh()
                neg_clip = neg_clip.arctanh()

            if cfg.reverse_aesthetic:
                lambda_aesthetic = -lambda_aesthetic

            loss_tv = total_variation_loss(images)
            loss_l1 = l1_loss(images)
            loss = (
                -lambda_aesthetic * aesthetic_score
                - cfg.lambda_clip * (pos_clip - neg_clip)
                + cfg.lambda_tv * loss_tv
                + cfg.lambda_l1 * loss_l1
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            if progress_callback:
                progress_callback(
                    step=step,
                    loss=loss.item(),
                    aesthetic_score=aesthetic_score.item() * 10,
                    pos_clip=pos_clip.item(),
                    neg_clip=neg_clip.item(),
                )

        self.image_stack = [i.detach().cpu() for i in image_stack]
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self, pos_texts, neg_texts):
        with torch.no_grad():
            image = torch.cat(
                [
                    interpolate(i, self.max_size, mode=self.mode).to(self.device)
                    for i in self.image_stack
                ]
            ).mean(0)
            image = image[..., self.pad : -self.pad, self.pad : -self.pad]
            image = image.tanh()
            image_np = inv_process(image.squeeze(0).detach().cpu())

            image = image.unsqueeze(0)
            if image.shape[-1] != self.full_size:
                image = interpolate(image, self.full_size, mode=self.mode)

            aesthetic_score, pos_clip_score, neg_clip_score = self.model.calc_scores(
                image, pos_texts, neg_texts
            )
            return (
                image_np,
                aesthetic_score * 10,
                pos_clip_score,
                neg_clip_score,
            )
