def linear_schedule(step, max_steps, start=0.5, end=0.0, **kwargs):
    return start + (end - start) * (step / max_steps)


def exponential_decay_schedule(step, max_steps, start=0.5, end=0.0, rate=0.1, **kwargs):
    return (start - end) * (1 - rate) ** step + end


def exponential_warmup_schedule(
    step, max_steps, start=0.5, end=0.0, rate=0.03, **kwargs
):
    initial_bias = (end - start) * (1 - rate) ** max_steps
    return (end - start) * (1 - rate) ** (max_steps - step) + start - initial_bias


schedule_map = {
    "linear": linear_schedule,
    "exponential_warmup": exponential_warmup_schedule,
    "exponential_decay": exponential_decay_schedule,
}
