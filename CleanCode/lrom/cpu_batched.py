"""CPU helpers for parallel batches of independent reduced systems."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ThreadedBatchedSolver:
    """Split a large stack of independent systems across persistent workers."""

    def __init__(self, workers: int = 8, minimum_batch: int = 50000):
        self.workers = int(workers)
        self.minimum_batch = int(minimum_batch)
        self.executor = ThreadPoolExecutor(max_workers=self.workers)

    @staticmethod
    def _solve_piece(matrices: np.ndarray, right_sides: np.ndarray) -> np.ndarray:
        return np.linalg.solve(matrices, right_sides[..., None])[..., 0]

    def solve(self, matrices: np.ndarray, right_sides: np.ndarray) -> np.ndarray:
        if len(matrices) < self.minimum_batch or self.workers == 1:
            return self._solve_piece(matrices, right_sides)
        boundaries = np.linspace(
            0, len(matrices), self.workers + 1, dtype=int
        )
        choices = [
            slice(start, stop)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
            if stop > start
        ]
        futures = [
            self.executor.submit(
                self._solve_piece, matrices[choice], right_sides[choice]
            )
            for choice in choices
        ]
        return np.concatenate([future.result() for future in futures])

    def close(self) -> None:
        self.executor.shutdown(wait=True)
