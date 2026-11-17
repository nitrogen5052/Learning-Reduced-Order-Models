"""Small, optional backends for the batched reduced linear solves.

The FOM, training, and scalar prediction paths use NumPy on the CPU.  Large
batches can optionally send only the small dense online systems to JAX.  This
keeps one public LROM implementation and makes the accelerator an optimization,
not a second scientific code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Runtime:
    """Resolved online runtime.

    Pass ``runtime.solve`` to ``ScatteringLROM.cross_sections``.  ``auto`` uses
    a JAX GPU when one is visible and otherwise returns the NumPy CPU runtime.
    """

    name: str
    device: str
    precision: str = "double"
    _solve_impl: object | None = None

    def solve(self, matrices: np.ndarray, right_sides: np.ndarray) -> np.ndarray:
        if self._solve_impl is None:
            return np.linalg.solve(matrices, right_sides[..., None])[..., 0]
        return np.asarray(self._solve_impl(matrices, right_sides))

    @property
    def accelerated(self) -> bool:
        return self.name == "jax-gpu"


def _jax_gpu_runtime(precision: str) -> Runtime:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "GPU mode requires a GPU-enabled JAX installation; see README.md"
        ) from exc

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX is installed, but no GPU device is visible")
    if precision not in {"single", "double"}:
        raise ValueError("precision must be 'single' or 'double'")
    jax.config.update("jax_enable_x64", precision == "double")
    dtype = jnp.complex64 if precision == "single" else jnp.complex128

    @jax.jit
    def solve(matrices, right_sides):
        a = jnp.asarray(matrices, dtype=dtype)
        b = jnp.asarray(right_sides, dtype=dtype)
        return jnp.linalg.solve(a, b[..., None])[..., 0]

    # Trigger compilation with a harmless representative system so notebook
    # timings do not accidentally include first-call compilation.
    solve(np.eye(2, dtype=complex)[None], np.ones((1, 2), complex)).block_until_ready()
    return Runtime("jax-gpu", str(devices[0]), precision, solve)


def get_runtime(backend: str = "cpu", precision: str = "double") -> Runtime:
    """Resolve ``cpu``, ``gpu``, or ``auto`` to one explicit runtime."""
    backend = backend.lower().strip()
    if backend == "cpu":
        return Runtime("numpy-cpu", "CPU", precision)
    if backend == "gpu":
        return _jax_gpu_runtime(precision)
    if backend == "auto":
        try:
            return _jax_gpu_runtime(precision)
        except RuntimeError:
            return Runtime("numpy-cpu", "CPU", precision)
    raise ValueError("backend must be 'cpu', 'gpu', or 'auto'")

