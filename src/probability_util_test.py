import math
from typing import Union

import numpy as np
import pytest

from probability_util import binary_search, log_binomial, log_factorial, least_squares_output_range, least_squares_slope_range, \
    binary_intercept, least_squares_through_point


@pytest.mark.parametrize(
    "arg,result",
    {
        0: 0,
        1: 0,
        2: math.log(2),
        3: math.log(2) + math.log(3),
        # These values were taken from wolfram alpha:
        10: 15.1044125730755152952257093292510,
        100: 363.73937555556349014407999336965,
        1000: 5912.128178488163348878130886725,
        10000: 82108.9278368143534553850300635,
        100000: 1051299.2218991218651292781082,
    }.items(),
)
def test_log_factorial(arg, result):
    np.testing.assert_allclose(log_factorial(arg), result, rtol=1e-5)


@pytest.mark.parametrize(
    "n,p,hits,result",
    [
        (1, 0.5, 0, np.log(0.5)),
        (1, 0.5, 1, np.log(0.5)),
        (1, 0.1, 0, np.log(0.9)),
        (1, 0.1, 1, np.log(0.1)),
        (2, [0, 1, 0.1, 0.5], 0, [0, -np.inf, np.log(0.9 ** 2), np.log(0.25)]),
        (2, [0, 1, 0.1, 0.5], 1, [-np.inf, -np.inf, np.log(0.1 * 0.9 * 2), np.log(0.5)]),
        (2, [0, 1, 0.1, 0.5], 2, [-np.inf, 0, np.log(0.1 ** 2), np.log(0.25)]),
        # Magic number comes from PDF[BinomialDistribution[10^10, 10^-6], 10000] on wolfram alpha.
        (10 ** 10, 10 ** -6, 10 ** 4, np.log(0.0039893915536591)),
        # Corner cases.
        (1, 0.0, 0, 0),
        (1, 0.0, 1, -np.inf),
        (1, 1.0, 0, -np.inf),
        (1, 1.0, 1, 0),
        # Array broadcast.
        (2, np.array([0.0, 0.5, 1.0]), 0, np.array([0.0, np.log(0.25), -np.inf])),
        (2, np.array([0.0, 0.5, 1.0]), 1, np.array([-np.inf, np.log(0.5), -np.inf])),
        (2, np.array([0.0, 0.5, 1.0]), 2, np.array([-np.inf, np.log(0.25), 0.0])),
    ],
)
def test_log_binomial(
    n: int, p: Union[float, np.ndarray], hits: int, result: Union[float, np.ndarray]
) -> None:
    np.testing.assert_allclose(log_binomial(n=n, p=p, hits=hits), result, rtol=1e-2)


def test_binary_search():
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=100.1) == 10
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=100) == 10
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=99.9) == 10
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=90) == 9
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=92) == 10
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=-100) == 0
    assert binary_search(func=lambda x: x**2, min_x=0, max_x=10**100, target=10**300) == 10**100


def test_least_squares_through_point():
    fit = least_squares_through_point(
        xs=np.array([1, 2, 3]),
        ys=np.array([2, 3, 4]),
        required_x=1,
        required_y=2)
    np.testing.assert_allclose(fit.slope, 1)
    np.testing.assert_allclose(fit.intercept, 1)

    fit = least_squares_through_point(
        xs=np.array([1, 2, 3]),
        ys=np.array([2, 3, 4]),
        required_x=1,
        required_y=1)
    np.testing.assert_allclose(fit.slope, 1.6, rtol=1e-5)
    np.testing.assert_allclose(fit.intercept, -0.6, atol=1e-5)


def test_binary_intercept():
    t = binary_intercept(func=lambda x: x**2, start_x=5, step=1, target_y=82.3, atol=0.01)
    assert t > 0 and abs(t**2 - 82.3) <= 0.01
    t = binary_intercept(func=lambda x: -x**2, start_x=5, step=1, target_y=-82.3, atol=0.01)
    assert t > 0 and abs(t**2 - 82.3) <= 0.01
    t = binary_intercept(func=lambda x: x**2, start_x=0, step=-1, target_y=82.3, atol=0.01)
    assert t < 0 and abs(t**2 - 82.3) <= 0.01
    t = binary_intercept(func=lambda x: -x**2, start_x=0, step=-1, target_y=-82.3, atol=0.2)
    assert t < 0 and abs(t**2 - 82.3) <= 0.2


def least_squares_output_range():
    low, high = least_squares_output_range(
        xs=[1, 2, 3],
        ys=[1, 5, 9],
        target_x=100,
        cost_increase=1,
    )
    assert 300 < low < 400 < high < 500


def test_least_squares_slope_range():
    low, mid, high = least_squares_slope_range(
        xs=[1, 2, 3],
        ys=[1, 5, 9],
        cost_increase=1,
    )
    np.testing.assert_allclose(mid, 4)
    assert 3 < low < 3.5 < mid < 4.5 < high < 5
