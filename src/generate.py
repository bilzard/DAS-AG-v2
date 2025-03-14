import hashlib
import os
from pathlib import Path

import polars as pl
from tqdm import tqdm

from das_ag_v2.config import Config, Prompt, load_yaml_to_dataclass
from das_ag_v2.trainer import DasGenerator
from das_ag_v2.util import seed_everything


def generate_one(aesthetic_predictor_path, clip_model_path, config, prompt):
    seed_everything(config.seed)
    generator = DasGenerator(aesthetic_predictor_path, clip_model_path, config=config)
    with tqdm(total=config.num_steps) as pbar:

        def update_progress_bar(step, loss, aesthetic_score, pos_clip, neg_clip):
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": loss,
                    "aesthetic_score": aesthetic_score,
                    "pos_clip_score": pos_clip,
                    "neg_clip_score": neg_clip,
                }
            )

        generator.generate(
            list(prompt.positive_prompts),
            list(prompt.negative_prompts),
            update_progress_bar,
        )

    result_img, aesthetic_score, pos_clip_score, neg_clip_score = generator.evaluate(
        prompt.positive_prompts, prompt.negative_prompts
    )
    return result_img, aesthetic_score, pos_clip_score, neg_clip_score


def main(num_samples_per_method=5):
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    AESTHETIC_PREDICTOR_PATH = os.environ.get("AESTHETIC_PREDICTOR_PATH")
    CLIP_MODEL_PATH = os.environ.get("CLIP_MODEL_PATH")

    assert PROJECT_ROOT is not None
    assert AESTHETIC_PREDICTOR_PATH is not None
    assert CLIP_MODEL_PATH is not None

    PROJECT_ROOT = Path(PROJECT_ROOT)

    output_path = PROJECT_ROOT / "data/generated_images_v2"
    output_path.mkdir(exist_ok=True)

    prompt = load_yaml_to_dataclass(
        f"{PROJECT_ROOT}/config/prompt/mona_lisa.yaml", Prompt
    )
    config_map = {
        "das_ag": load_yaml_to_dataclass(f"{PROJECT_ROOT}/config/das_ag.yaml", Config),
        "das": load_yaml_to_dataclass(f"{PROJECT_ROOT}/config/das.yaml", Config),
        "das_rag": load_yaml_to_dataclass(
            f"{PROJECT_ROOT}/config/das_rag.yaml", Config
        ),
        "das_ag_plus": load_yaml_to_dataclass(
            f"{PROJECT_ROOT}/config/das_ag_plus.yaml", Config
        ),
        "das_rag_plus": load_yaml_to_dataclass(
            f"{PROJECT_ROOT}/config/das_rag_plus.yaml", Config
        ),
    }

    file_hashes = []
    methods = []
    aesthetic_scores = []
    pos_clip_scores = []
    neg_clip_scores = []

    for _ in range(num_samples_per_method):
        for method, config in config_map.items():
            print(f"Generating for {method}")
            print(f"Aesthetic predictor path: {AESTHETIC_PREDICTOR_PATH}")
            print(f"CLIP model path: {CLIP_MODEL_PATH}")

            try:
                image, aesthetic_score, pos_clip_score, neg_clip_score = generate_one(
                    AESTHETIC_PREDICTOR_PATH,
                    CLIP_MODEL_PATH,
                    config,
                    prompt,
                )
                file_hash = hashlib.sha256(image.tobytes()).hexdigest()
                output_file = output_path / f"{file_hash}.png"
                print(f"Saving to {output_file}")
                image.save(output_file)

                methods.append(method)
                file_hashes.append(file_hash)
                aesthetic_scores.append(aesthetic_score)
                pos_clip_scores.append(pos_clip_score)
                neg_clip_scores.append(neg_clip_score)

            except Exception as e:
                print(f"Failed to generate for {method}")
                print(e)
                continue

    df = pl.DataFrame(
        {
            "method": methods,
            "file_hash": file_hashes,
            "aesthetic_score": aesthetic_scores,
            "pos_clip_score": pos_clip_scores,
            "neg_clip_score": neg_clip_scores,
        }
    )
    df.write_csv(f"{PROJECT_ROOT}/data/score_v2.csv")


if __name__ == "__main__":
    main()
