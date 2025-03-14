import os

import streamlit as st

from das_ag_v2.config import Config, Prompt, load_yaml_to_dataclass
from das_ag_v2.schedule import schedule_map
from das_ag_v2.trainer import DasGenerator
from das_ag_v2.util import is_deterministic_algorithm_enabled, seed_everything


def main():
    assert CLIP_MODEL_PATH is not None
    assert AESTHETIC_PREDICTOR_PATH is not None

    prompt = load_yaml_to_dataclass("config/prompt/mona_lisa.yaml", Prompt)

    st.title("Aesthetic Image Generation with CLIP")

    # Positive Prompts
    st.header("Positive Prompts")
    if "positive_prompts" not in st.session_state:
        st.session_state.positive_prompts = list(prompt.positive_prompts)

    def add_pos_input():
        st.session_state.positive_prompts.append("")

    def remove_pos_input():
        if len(st.session_state.positive_prompts) > 0:
            st.session_state.positive_prompts.pop()

    for i in range(len(st.session_state.positive_prompts)):
        st.session_state.positive_prompts[i] = st.text_area(
            f"Prompt {i + 1}", st.session_state.positive_prompts[i]
        )
    col1, col2 = st.columns(2)
    with col1:
        st.button("Add", on_click=add_pos_input, key="add_pos")
    with col2:
        st.button("Remove", on_click=remove_pos_input, key="remove_pos")

    # Negative Prompts
    st.header("Negative Prompts")
    if "negative_prompts" not in st.session_state:
        st.session_state.negative_prompts = list(prompt.negative_prompts)

    def add_neg_input():
        st.session_state.negative_prompts.append("")

    def remove_neg_input():
        if len(st.session_state.negative_prompts) > 0:
            st.session_state.negative_prompts.pop()

    for i in range(len(st.session_state.negative_prompts)):
        st.session_state.negative_prompts[i] = st.text_area(
            f"Prompt {i + 1}", st.session_state.negative_prompts[i]
        )
    col1, col2 = st.columns(2)
    with col1:
        st.button("Add", on_click=add_neg_input, key="add_neg")
    with col2:
        st.button("Remove", on_click=remove_neg_input, key="remove_neg")

    # Sidebar
    st.sidebar.header("General")
    resolution = st.sidebar.selectbox(
        "Resolution", [384, 768], index=1, help="Resolution of the generated image"
    )
    num_steps = st.sidebar.slider(
        "#steps", min_value=0, max_value=200, value=100, step=10
    )
    batch_size = st.sidebar.slider(
        "batch size", min_value=4, max_value=32, value=8, step=4
    )
    lr = st.sidebar.slider("lr", min_value=0.0, max_value=0.2, value=0.10, step=0.01)

    st.sidebar.header("CLIP")
    clip_weight = st.sidebar.slider(
        "CLIP Weight",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Weight of CLIP loss",
    )
    nonlinear_loss_scaling = st.sidebar.checkbox(
        "Nonlinear Loss Scaling",
        value=True,
        help="If enabled, the loss will be scaled nonlinearly",
    )

    st.sidebar.header("Aesthetic")
    reverse_aesthetic = st.sidebar.checkbox(
        "Reverse Aesthetic",
        value=False,
        help="If enabled, the aesthetic score will be reversed",
    )
    aesthetic_range = st.sidebar.slider(
        "Aesthetic Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.1,
    )
    aesthetic_schedule = st.sidebar.selectbox(
        "Aesthetic Schedule",
        list(schedule_map.keys()),
        index=2,
    )
    aesthetic_decay_rate = st.sidebar.slider(
        "Aesthetic Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        disabled=aesthetic_schedule == "linear",
    )

    st.sidebar.header("Generation")
    use_deterministic_algorithm = (
        st.sidebar.checkbox(
            "Use Deterministic Algorithm",
            value=False,
            help="**Note**: it does not ensure reproducibility and it becomes slower",
        )
        if is_deterministic_algorithm_enabled()
        else False
    )
    interpolation_mode = st.sidebar.selectbox(
        "Interpolation Mode",
        ["bilinear", "bicubic", "nearest"]
        if not use_deterministic_algorithm
        else ["bilinear"],
    )
    set_seed = st.sidebar.checkbox("Set Seed", value=False)
    seed = st.sidebar.number_input("seed", value=0, step=1, disabled=not set_seed)

    st.sidebar.header("Regularization")
    lambda_tv_exp = st.sidebar.slider(
        "Texture Suppression (TV)",
        min_value=-8,
        max_value=0,
        value=-8,
        help="higher value suppresses texture and increases smoothness",
    )
    lambda_tv = 10**lambda_tv_exp
    lambda_l1 = st.sidebar.slider(
        "Color Suppression (L1)",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.05,
        help="higher value suppresses color and increases grayness",
    )

    st.sidebar.header("Augmentation")
    apply_augmentation = st.sidebar.checkbox("Apply Augmentation", value=True)
    st.sidebar.subheader("Gaussian Noise")
    noise_schedule = st.sidebar.selectbox(
        "Noise Schedule",
        list(schedule_map.keys()),
        index=1,
        disabled=not apply_augmentation,
    )
    noise_stds = st.sidebar.slider(
        "Noise Intensity Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.2, 0.5),
        step=0.05,
        disabled=not apply_augmentation,
    )
    noise_decay_rate = st.sidebar.slider(
        "Noise Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        disabled=not apply_augmentation or noise_schedule == "linear",
    )
    st.sidebar.subheader("Color Shift")
    color_shift_schedule = st.sidebar.selectbox(
        "Color Shift Schedule",
        list(schedule_map.keys()),
        index=1,
        disabled=not apply_augmentation,
    )
    color_shift_range = st.sidebar.slider(
        "Color Shift Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.05, 0.30),
        step=0.05,
        disabled=not apply_augmentation,
    )
    color_shift_decay_rate = st.sidebar.slider(
        "Color Shift Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        disabled=not apply_augmentation or color_shift_schedule == "linear",
    )
    st.sidebar.subheader("Positional Jitter")
    pos_jitter_schedule = st.sidebar.selectbox(
        "Positional Jitter Schedule",
        list(schedule_map.keys()),
        index=1,
        disabled=not apply_augmentation,
    )
    pos_jitter_decay_rate = st.sidebar.slider(
        "Positional Jitter Decay Rate",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        disabled=not apply_augmentation or pos_jitter_schedule == "linear",
    )

    if resolution == 384:
        image_resolutions = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        max_shift = 64
    elif resolution == 768:
        image_resolutions = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        max_shift = 128
    else:
        raise ValueError("Invalid resolution")

    cfg = Config(
        num_steps=num_steps,
        batch_size=batch_size,
        lambda_tv=lambda_tv,
        lambda_l1=lambda_l1,
        lr=lr,
        checkpoint_interval=num_steps,
        noise_schedule=noise_schedule,
        noise_std_range=noise_stds,
        noise_decay_rate=noise_decay_rate,
        color_shift_schedule=color_shift_schedule,
        color_shift_decay_rate=color_shift_decay_rate,
        color_shift_range=color_shift_range,
        seed=seed,
        mode=interpolation_mode,
        use_deterministic_algorithm=use_deterministic_algorithm,
        apply_augmentation=apply_augmentation,
        lambda_clip=clip_weight,
        aesthetic_range=aesthetic_range,
        aesthetic_schedule=aesthetic_schedule,
        aesthetic_decay_rate=aesthetic_decay_rate,
        reverse_aesthetic=reverse_aesthetic,
        image_resolutions=image_resolutions,
        max_shift=max_shift,
        pos_jitter_schedule=pos_jitter_schedule,
        pos_jitter_decay_rate=pos_jitter_decay_rate,
        nonlinear_loss_scaling=nonlinear_loss_scaling,
    )
    positive_prompts = st.session_state.positive_prompts
    negative_prompts = st.session_state.negative_prompts

    if st.button("Generate Image"):
        if len(positive_prompts) > 0:
            st.write("**Positive Prompts**:")
            for pos_prompt in positive_prompts:
                st.write(f"`{pos_prompt}`")

        if len(negative_prompts) > 0:
            st.write("**Negative Prompts**:")
            for neg_prompt in negative_prompts:
                st.write(f"`{neg_prompt}`")
            st.write("Generating Image...")

        if set_seed:
            seed_everything(
                cfg.seed,
                use_deterministic_algorithm=cfg.use_deterministic_algorithm,
            )
        attacker = DasGenerator(AESTHETIC_PREDICTOR_PATH, CLIP_MODEL_PATH, config=cfg)
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(step, **kwargs):
            aesthetic_score = kwargs.get("aesthetic_score", 0.0)
            pos_clip = kwargs.get("pos_clip", 0.0)
            neg_clip = kwargs.get("neg_clip", 0.0)
            progress = (step + 1) / cfg.num_steps
            progress_bar.progress(progress)
            status_text.markdown(
                f"**step: {step + 1}/{cfg.num_steps}**, loss: {kwargs['loss']:.3f}, aesthetic score: {aesthetic_score:.2f}, CLIP score: {pos_clip:.3f} (negative: {neg_clip:.3f})"
            )

        attacker.generate(
            positive_prompts, negative_prompts, progress_callback=update_progress
        )
        st.success("Image Generation Completed!")

        st.subheader("Result")
        result_img, aesthetic_score, pos_clip_score, neg_clip_score = attacker.evaluate(
            positive_prompts, negative_prompts
        )
        st.image(
            result_img,
            caption=f"Aesthetic Score: {aesthetic_score.item():.2f}, CLIP Score: {pos_clip_score.item():.3f} (Negative: {neg_clip_score.item():.3f})",
            use_container_width=True,
            output_format="PNG",
        )


if __name__ == "__main__":
    AESTHETIC_PREDICTOR_PATH = os.environ.get("AESTHETIC_PREDICTOR_PATH")
    CLIP_MODEL_PATH = os.environ.get("CLIP_MODEL_PATH")
    main()
