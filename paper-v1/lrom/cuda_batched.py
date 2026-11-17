"""Optional Windows CUDA backend for small complex batched linear solves.

This module intentionally has no CuPy or PyTorch dependency.  It calls the
CUDA runtime and cuBLAS libraries distributed by NVIDIA's Python toolkit
packages through ``ctypes``.  The narrow interface keeps GPU acceleration
optional and makes it straightforward to compare against ``numpy.linalg``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np


class CudaError(RuntimeError):
    """Raised when a CUDA runtime or cuBLAS call fails."""


def _check(code: int, operation: str) -> None:
    if code != 0:
        raise CudaError(f"{operation} failed with status {code}")


class CudaBatchedSolver:
    """Solve batches of square complex128 systems with cuBLAS LU."""

    def __init__(self, library_directory: str | Path | None = None):
        if sys.platform != "win32":
            raise RuntimeError("the current direct CUDA loader targets Windows")
        if library_directory is None:
            library_directory = (
                Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" /
                "cu13" / "bin" / "x86_64"
            )
        directory = Path(library_directory).resolve()
        self.runtime = ctypes.WinDLL(str(directory / "cudart64_13.dll"))
        self.cublas = ctypes.WinDLL(str(directory / "cublas64_13.dll"))
        self._configure_signatures()
        self.handle = ctypes.c_void_p()
        _check(self.cublas.cublasCreate_v2(ctypes.byref(self.handle)), "cublasCreate")

    def _configure_signatures(self) -> None:
        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        int_p = ctypes.POINTER(ctypes.c_int)
        self.runtime.cudaMalloc.argtypes = [ctypes.POINTER(void_p), size_t]
        self.runtime.cudaMalloc.restype = ctypes.c_int
        self.runtime.cudaFree.argtypes = [void_p]
        self.runtime.cudaFree.restype = ctypes.c_int
        self.runtime.cudaMemcpy.argtypes = [void_p, void_p, size_t, ctypes.c_int]
        self.runtime.cudaMemcpy.restype = ctypes.c_int
        self.runtime.cudaDeviceSynchronize.argtypes = []
        self.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
        self.cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(void_p)]
        self.cublas.cublasCreate_v2.restype = ctypes.c_int
        self.cublas.cublasDestroy_v2.argtypes = [void_p]
        self.cublas.cublasDestroy_v2.restype = ctypes.c_int
        self.cublas.cublasZgetrfBatched.argtypes = [
            void_p, ctypes.c_int, void_p, ctypes.c_int, void_p, void_p,
            ctypes.c_int,
        ]
        self.cublas.cublasZgetrfBatched.restype = ctypes.c_int
        self.cublas.cublasZgetrsBatched.argtypes = [
            void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, void_p,
            ctypes.c_int, void_p, void_p, ctypes.c_int, int_p, ctypes.c_int,
        ]
        self.cublas.cublasZgetrsBatched.restype = ctypes.c_int
        self.cublas.cublasCgetrfBatched.argtypes = (
            self.cublas.cublasZgetrfBatched.argtypes
        )
        self.cublas.cublasCgetrfBatched.restype = ctypes.c_int
        self.cublas.cublasCgetrsBatched.argtypes = (
            self.cublas.cublasZgetrsBatched.argtypes
        )
        self.cublas.cublasCgetrsBatched.restype = ctypes.c_int

    def _malloc(self, byte_count: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        _check(self.runtime.cudaMalloc(ctypes.byref(pointer), byte_count), "cudaMalloc")
        return pointer

    def _copy_to_device(self, destination: ctypes.c_void_p, array: np.ndarray) -> None:
        _check(
            self.runtime.cudaMemcpy(
                destination, ctypes.c_void_p(array.ctypes.data), array.nbytes, 1
            ),
            "cudaMemcpy host-to-device",
        )

    def solve(
        self,
        matrices: np.ndarray,
        right_sides: np.ndarray,
        dtype=np.complex128,
    ) -> np.ndarray:
        """Return solutions for ``matrices @ x = right_sides``.

        ``matrices`` has shape ``(batch, n, n)`` and ``right_sides`` may have
        shape ``(batch, n)`` or ``(batch, n, nrhs)``.
        """
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError("CUDA batched solve supports complex64 or complex128")
        matrices = np.asarray(matrices, dtype=dtype)
        right_sides = np.asarray(right_sides, dtype=dtype)
        vector_input = right_sides.ndim == 2
        if vector_input:
            right_sides = right_sides[..., None]
        if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
            raise ValueError("matrices must have shape (batch, n, n)")
        batch, size, _ = matrices.shape
        if right_sides.shape[:2] != (batch, size):
            raise ValueError("right sides do not match the matrix batch")
        nrhs = right_sides.shape[2]

        # A contiguous transpose is the column-major representation cuBLAS
        # expects for each matrix in the batch.
        matrix_storage = np.ascontiguousarray(matrices.swapaxes(-1, -2))
        rhs_storage = np.ascontiguousarray(right_sides.swapaxes(-1, -2))
        matrix_bytes = matrix_storage.nbytes
        rhs_bytes = rhs_storage.nbytes
        device_allocations: list[ctypes.c_void_p] = []
        try:
            device_a = self._malloc(matrix_bytes)
            device_allocations.append(device_a)
            device_b = self._malloc(rhs_bytes)
            device_allocations.append(device_b)
            pivots = self._malloc(batch * size * np.dtype(np.int32).itemsize)
            device_allocations.append(pivots)
            information = self._malloc(batch * np.dtype(np.int32).itemsize)
            device_allocations.append(information)

            itemsize = dtype.itemsize
            matrix_pointers = (
                np.uint64(device_a.value) +
                np.arange(batch, dtype=np.uint64) * size * size * itemsize
            )
            rhs_pointers = (
                np.uint64(device_b.value) +
                np.arange(batch, dtype=np.uint64) * size * nrhs * itemsize
            )
            device_matrix_pointers = self._malloc(matrix_pointers.nbytes)
            device_allocations.append(device_matrix_pointers)
            device_rhs_pointers = self._malloc(rhs_pointers.nbytes)
            device_allocations.append(device_rhs_pointers)
            self._copy_to_device(device_a, matrix_storage)
            self._copy_to_device(device_b, rhs_storage)
            self._copy_to_device(device_matrix_pointers, matrix_pointers)
            self._copy_to_device(device_rhs_pointers, rhs_pointers)

            getrf = (
                self.cublas.cublasCgetrfBatched
                if dtype == np.dtype(np.complex64)
                else self.cublas.cublasZgetrfBatched
            )
            getrs = (
                self.cublas.cublasCgetrsBatched
                if dtype == np.dtype(np.complex64)
                else self.cublas.cublasZgetrsBatched
            )
            _check(
                getrf(
                    self.handle, size, device_matrix_pointers, size, pivots,
                    information, batch,
                ),
                "cublasZgetrfBatched",
            )
            host_information = ctypes.c_int()
            _check(
                getrs(
                    self.handle, 0, size, nrhs, device_matrix_pointers, size,
                    pivots, device_rhs_pointers, size,
                    ctypes.byref(host_information), batch,
                ),
                "cublasZgetrsBatched",
            )
            if host_information.value:
                raise CudaError(
                    f"cublasZgetrsBatched reported argument {host_information.value}"
                )
            _check(self.runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
            output_storage = np.empty_like(rhs_storage)
            _check(
                self.runtime.cudaMemcpy(
                    ctypes.c_void_p(output_storage.ctypes.data), device_b,
                    output_storage.nbytes, 2,
                ),
                "cudaMemcpy device-to-host",
            )
            output = output_storage.swapaxes(-1, -2)
            return output[..., 0] if vector_input else output
        finally:
            for pointer in reversed(device_allocations):
                self.runtime.cudaFree(pointer)

    def solve_single_precision(
        self, matrices: np.ndarray, right_sides: np.ndarray
    ) -> np.ndarray:
        """Solve in complex64 and return complex128-compatible coordinates."""
        return self.solve(matrices, right_sides, dtype=np.complex64).astype(
            np.complex128
        )

    def solve_mixed_precision(
        self,
        matrices: np.ndarray,
        right_sides: np.ndarray,
        refinement_steps: int = 1,
    ) -> np.ndarray:
        """Use complex64 GPU solves with complex128 residual refinement."""
        matrices = np.asarray(matrices, dtype=np.complex128)
        right_sides = np.asarray(right_sides, dtype=np.complex128)
        solution = self.solve_single_precision(matrices, right_sides)
        for _ in range(int(refinement_steps)):
            residual = right_sides - np.einsum(
                "bij,bj->bi", matrices, solution, optimize=False
            )
            solution += self.solve_single_precision(matrices, residual)
        return solution

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.cublas.cublasDestroy_v2(self.handle)
            self.handle = None

    def __enter__(self) -> "CudaBatchedSolver":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
