"""Analytic isotropic checks and contributor-stop edge cases for the CPU oracle."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.affine_rays_cpu import project_model, ray_parameters
from research.median_solver import solve


def main():
    model = (np.array([[0., 0., 2.]]), np.array([[.01, .01, .01]]),
             np.array([.9]), np.eye(3)[None])
    camera = dict(width=96, height=96, fx=100., fy=100., position=[0., 0., 0.], rotation=np.eye(3).tolist())
    projected = project_model(model, camera, resolution=1)
    np.testing.assert_allclose(projected["conic"], [[.04, 0, .04]], atol=1e-14)
    np.testing.assert_allclose(projected["plane"], [[0, 0, 2]], atol=1e-14)
    np.testing.assert_allclose(projected["widths"], [.1], atol=1e-14)
    a, p, w, init = ray_parameters(projected, [48, 48])
    np.testing.assert_allclose(a, [.9 * np.exp(-.01)], atol=1e-14)
    np.testing.assert_allclose(p, [2.], atol=1e-14)
    assert init == 2.
    oracle = 2 - .1 * np.sqrt(-2 * np.log(.75 / a[0]))
    assert abs(solve(a, p, w)["depth"] - oracle) < 1e-9
    # Four central .99 primitives: keep two, exclude the one that would
    # put final T strictly below 1e-4 (float64 rounding is intentionally tested).
    many = {key: np.repeat(value, 4, axis=0) if isinstance(value, np.ndarray) else value
            for key, value in projected.items()}
    many["opacity"] = np.full(4, .99)
    a, p, w, init = ray_parameters(many, [47.5, 47.5])
    assert len(a) == 2, len(a)
    assert np.prod(1 - a) >= 1e-4
    assert np.prod(1 - np.r_[a, .99]) < 1e-4
    print("Analytic affine projection, median, and contributor early-stop checks passed.")
    print("Full anisotropic CUDA parity remains pending.")


if __name__ == "__main__":
    main()
