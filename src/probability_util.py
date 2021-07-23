import math
from typing import Union, Callable

import numpy as np


def log_binomial(*, p: Union[float, np.ndarray], n: int, hits: int) -> Union[float, np.ndarray]:
    r"""Approximates $\ln(P(hits = B(n, p)))$; the natural logarithm of a binomial distribution.

    All computations are done in log space to ensure intermediate values can be represented as
    floating point numbers without underflowing to 0 or overflowing to infinity. This is necessary
    when computing likelihoods over many samples. For example, if 80% of a million samples are hits,
    the maximum likelihood estimate is p=0.8. But even this optimal estimate assigns a prior
    probability of roughly 10^-217322 for seeing *exactly* 80% hits out of a million (whereas the
    smallest representable double is roughly 10^-324).

    This method can be broadcast over multiple hypothesis probabilities.

    Args:
        p: The independent probability of a hit occurring for each sample. This can also be an array
            of probabilities, in which case the function is broadcast over the array.
        n: The number of samples that were taken.
        hits: The number of hits that were observed amongst the samples that were taken.

    Returns:
        $\ln(P(hits = B(n, p)))$
    """
    # Clamp probabilities into the valid [0, 1] range (in case float error put them outside it).
    p_clipped = np.clip(p, 0, 1)

    result = np.zeros(shape=p_clipped.shape, dtype=np.float32)
    misses = n - hits

    # Handle p=0 and p=1 cases separately, to avoid arithmetic warnings.
    if hits:
        result[p_clipped == 0] = -np.inf
    if misses:
        result[p_clipped == 1] = -np.inf

    # Multiply p**hits and (1-p)**misses onto the total, in log space.
    result[p_clipped != 0] += np.log(p_clipped[p_clipped != 0]) * hits
    result[p_clipped != 1] += np.log1p(-p_clipped[p_clipped != 1]) * misses

    # Multiply (n choose hits) onto the total, in log space.
    log_n_choose_hits = log_factorial(n) - log_factorial(misses) - log_factorial(hits)
    result += log_n_choose_hits

    return result


def log_factorial(n: int) -> float:
    r"""Approximates $\ln(n!)$; the natural logarithm of a factorial.

    Uses Stirling's approximation for large n.
    """
    if n < 20:
        return sum(math.log(k) for k in range(1, n + 1))
    return (n + 0.5) * math.log(n) - n + math.log(2 * np.pi) / 2


def binary_search(*, func: Callable[[int], float], min_x: int, max_x: int, target: float) -> int:
    """Performs an approximate granular binary search over a monotonically ascending function."""
    while max_x > min_x + 1:
        med_x = (min_x + max_x) // 2
        out = func(med_x)
        if out < target:
            min_x = med_x
        elif out > target:
            max_x = med_x
        else:
            return med_x
    fmax = func(max_x)
    fmin = func(min_x)
    dmax = 0 if fmax == target else fmax - target
    dmin = 0 if fmin == target else fmin - target
    return max_x if abs(dmax) < abs(dmin) else min_x
