import torch
import torch.nn as nn
import torchvision.transforms as T
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from transformers import AutoProcessor, SiglipTextModel


def inv_process(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    inv_process = T.Compose(
        [
            T.Normalize(
                mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
            ),
            T.ToPILImage(),
        ]
    )
    return inv_process(image)


class AestheticModelV2(nn.Module):
    def __init__(self, predictor_path, clip_model_path):
        super().__init__()

        self.predictor_path = predictor_path
        self.clip_model_path = clip_model_path
        self.predictor, self.text_encoder, self.processor = self.load_()
        self.pos_text_feature = None
        self.neg_text_feature = None

    def load_(self):
        predictor, _ = convert_v2_5_from_siglip(
            self.predictor_path,
            encoder_model_name=self.clip_model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )

        text_encoder = SiglipTextModel.from_pretrained(
            self.clip_model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )

        processor = AutoProcessor.from_pretrained(self.clip_model_path)

        return predictor, text_encoder, processor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def calc_scores(self, image, pos_texts: list[str], neg_texts: list[str]):
        output = self.predictor(image)
        image_features = self.normalize(output.hidden_states)

        if len(pos_texts) > 0:
            if self.pos_text_feature is None:
                inputs = self.processor.tokenizer(
                    pos_texts, padding="max_length", return_tensors="pt"
                )
                pos_text_features = self.normalize(
                    self.text_encoder(**inputs.to(image.device)).pooler_output
                )
                self.pos_text_feature = pos_text_features
            else:
                pos_text_features = self.pos_text_feature
            pos_clip_score = (image_features @ pos_text_features.T).mean()
        else:
            pos_clip_score = torch.zeros(1, device=image.device)

        if len(neg_texts) > 0:
            if self.neg_text_feature is None:
                inputs = self.processor.tokenizer(
                    neg_texts, padding="max_length", return_tensors="pt"
                )
                neg_text_features = self.normalize(
                    self.text_encoder(**inputs.to(image.device)).pooler_output
                )
                self.neg_text_feature = neg_text_features
            else:
                neg_text_features = self.neg_text_feature

            neg_clip_score = (image_features @ neg_text_features.T).mean()
        else:
            neg_clip_score = torch.zeros(1, device=image.device)

        aesthetic_score = output.logits.mean() / 10.0
        return aesthetic_score, pos_clip_score, neg_clip_score
