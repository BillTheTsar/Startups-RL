"""Apply training performance settings without changing learning state.

Unlike TrainingConfig, this module imports PyTorch. The thread count applies
to the entire process, including CPU neural opponents and CPU tensor operations.
It does not control Python workers, game concurrency, or CUDA thread blocks.
"""

from functools import wraps

import torch

from .config import _positive_integer


def configure_cpu_threads(count: int) -> int:
    """Set PyTorch's intra-operation CPU threads and return the effective count.

    Call before model loading or training work. A positive integer is required;
    reject invalid values before inspecting or changing PyTorch state. Repeated
    calls with the current value do nothing, and callers can select a different
    count for later runs in the same process. No RNG, weights or optimizer state
    is changed.

    Inter-operation threads are left alone. That separate setting is useful for
    parallel execution graphs, is not used by this eager feed-forward pipeline,
    and cannot safely be reset for each continuation in a shared process.
    """
    _positive_integer(count, "cpu_threads")
    if torch.get_num_threads() != count:
        torch.set_num_threads(count)
    return torch.get_num_threads()


def restore_cpu_threads_on_error(function):
    """Keep a setup function's selected thread count only when setup succeeds.

    Restoring the caller's count after an error prevents a rejected configuration
    or checkpoint from changing later work in the same process. Use on synchronous
    setup functions, not an active training run or its interruption handling.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = torch.get_num_threads()
        try:
            return function(*args, **kwargs)
        except BaseException:
            configure_cpu_threads(previous)
            raise

    return wrapped
