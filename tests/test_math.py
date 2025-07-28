import random
import math
import pytest
from kesslergame.math_utils import solve_quadratic

def approx_equal(a: float, b: float, tol: float = 1e-8) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)

def test_solve_quadratic_known_roots():
    random.seed(0)
    for _ in range(1_000_000):
        # Random real roots
        r1 = random.uniform(-1000, 1000)
        r2 = random.uniform(-1000, 1000)

        # Form coefficients from (x - r1)(x - r2)
        a = 1.0
        b = -(r1 + r2)
        c = r1 * r2

        # Solve
        x0, x1 = solve_quadratic(a, b, c)

        # Compare sorted
        expected_roots = sorted([r1, r2])
        actual_roots = sorted([x0, x1])

        assert approx_equal(actual_roots[0], expected_roots[0]), f"Expected {expected_roots}, got {actual_roots}"
        assert approx_equal(actual_roots[1], expected_roots[1]), f"Expected {expected_roots}, got {actual_roots}"

def test_no_real_roots_returns_nan():
    x0, x1 = solve_quadratic(1, 0, 1)
    assert math.isnan(x0) and math.isnan(x1)

def test_linear_case():
    x0, x1 = solve_quadratic(0, 2, -4)
    assert approx_equal(x0, 2.0)
    assert approx_equal(x1, 2.0)

def test_all_zero_case():
    x0, x1 = solve_quadratic(0, 0, 0)
    assert approx_equal(x0, 0.0)
    assert approx_equal(x1, 0.0)

def test_linear_unsolvable():
    x0, x1 = solve_quadratic(0, 0, 5)
    assert math.isnan(x0) and math.isnan(x1)
