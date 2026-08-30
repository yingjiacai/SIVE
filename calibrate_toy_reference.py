"""Numerically calibrate the continuous Gaussian-localized u^2 v^2 target.

The two-dimensional integral is reduced analytically over ``u`` and evaluated
as a one-dimensional adaptive quadrature over ``v``.  This script does not run
SGLD and therefore separates the continuous finite-temperature reference from
finite-step and finite-chain effects.
"""

import argparse
import json
import math


def _simpson(function, left, right, f_left, f_mid, f_right):
    return (right - left) * (f_left + 4.0 * f_mid + f_right) / 6.0


def _adaptive_simpson(function, left, right, tolerance, max_depth=30):
    midpoint = (left + right) / 2.0
    f_left = function(left)
    f_mid = function(midpoint)
    f_right = function(right)
    whole = _simpson(function, left, right, f_left, f_mid, f_right)

    def recurse(a, b, fa, fm, fb, estimate, eps, depth):
        c = (a + b) / 2.0
        left_mid = (a + c) / 2.0
        right_mid = (c + b) / 2.0
        f_left_mid = function(left_mid)
        f_right_mid = function(right_mid)
        left_estimate = _simpson(function, a, c, fa, f_left_mid, fm)
        right_estimate = _simpson(function, c, b, fm, f_right_mid, fb)
        correction = left_estimate + right_estimate - estimate
        if depth <= 0 or abs(correction) <= 15.0 * eps:
            return left_estimate + right_estimate + correction / 15.0
        return recurse(
            a, c, fa, f_left_mid, fm, left_estimate, eps / 2.0, depth - 1
        ) + recurse(
            c, b, fm, f_right_mid, fb, right_estimate, eps / 2.0, depth - 1
        )

    return recurse(
        left, right, f_left, f_mid, f_right, whole, tolerance, max_depth
    )


def _integrate_real_line(function, center_v, h, tolerance):
    lower = center_v - 12.0 * h
    upper = center_v + 12.0 * h
    landmarks = [lower, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, upper]
    points = sorted({min(upper, max(lower, point)) for point in landmarks})
    intervals = [(a, b) for a, b in zip(points[:-1], points[1:]) if b > a]
    return sum(
        _adaptive_simpson(function, a, b, tolerance / len(intervals))
        for a, b in intervals
    )


def localized_uv2_targets(t=10000.0, h=2.0, center_u=0.2, center_v=0.2,
                          tolerance=1e-18):
    """Return ``t E[K]`` and ``t^2 Var[K]`` for ``K(u,v)=u^2 v^2``."""
    if t <= 0 or h <= 0:
        raise ValueError("t and h must be positive.")

    normalizer_v = 1.0 / (math.sqrt(2.0 * math.pi) * h)

    def components(v):
        gaussian_v = normalizer_v * math.exp(-0.5 * ((v - center_v) / h) ** 2)
        denominator = 1.0 + 2.0 * t * (v ** 2) * (h ** 2)
        tilted_variance = (h ** 2) / denominator
        tilted_mean = center_u / denominator
        integrated_weight = math.exp(
            -t * (v ** 2) * (center_u ** 2) / denominator
        ) / math.sqrt(denominator)
        common = gaussian_v * integrated_weight
        u2 = tilted_variance + tilted_mean ** 2
        u4 = (
            3.0 * tilted_variance ** 2
            + 6.0 * tilted_variance * tilted_mean ** 2
            + tilted_mean ** 4
        )
        return common, common * (v ** 2) * u2, common * (v ** 4) * u4

    partition = _integrate_real_line(lambda v: components(v)[0], center_v, h, tolerance)
    first = _integrate_real_line(lambda v: components(v)[1], center_v, h, tolerance)
    second = _integrate_real_line(lambda v: components(v)[2], center_v, h, tolerance)
    mean_k = first / partition
    variance_k = second / partition - mean_k ** 2
    return {
        "t": t,
        "h": h,
        "center": [center_u, center_v],
        "partition_relative_to_gaussian": partition,
        "t_E_K": t * mean_k,
        "t2_Var_K": (t ** 2) * variance_k,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=float, default=10000.0)
    parser.add_argument("--h", type=float, default=2.0)
    parser.add_argument("--center-u", type=float, default=0.2)
    parser.add_argument("--center-v", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=1e-18)
    args = parser.parse_args()
    result = localized_uv2_targets(
        t=args.t,
        h=args.h,
        center_u=args.center_u,
        center_v=args.center_v,
        tolerance=args.tolerance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
