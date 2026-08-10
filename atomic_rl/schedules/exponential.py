import math


def get_exponential_schedule(
    step: int, start_val: float, end_val: float, decay_rate: float
) -> float:
    """
    Exponentially decays a value, decay rate controls how fast it drops.

    Args:
        step (int): The current step.
        start_val (float): The starting value.
        end_val (float): The ending value.
        decay_rate (float): The decay rate.

    Returns:
        float: The scheduled value at the current step.
    """
    return end_val + (start_val - end_val) * math.exp(-1.0 * step / decay_rate)
