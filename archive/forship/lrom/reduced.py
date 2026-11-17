"""Centered reduced bases and the learned implicit reduced equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def latin_hypercube(lower: np.ndarray, upper: np.ndarray, count: int, seed: int) -> np.ndarray:
    """A compact Latin-hypercube sampler with deterministic NumPy randomness."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    unit = np.empty((count, lower.size))
    for column in range(lower.size):
        unit[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return lower + unit * (upper - lower)


def _trapezoid_sqrt_weights(radius: np.ndarray) -> np.ndarray:
    widths = np.diff(radius)
    weights = np.empty_like(radius)
    weights[0], weights[-1] = widths[0] / 2, widths[-1] / 2
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2
    return np.sqrt(weights)


@dataclass
class ReducedBasis:
    """A central reference wavefunction plus SVD modes of its deviations."""

    reference: np.ndarray
    vectors: np.ndarray
    radius: np.ndarray
    singular_values: np.ndarray

    @classmethod
    def fit(cls, reference: np.ndarray, snapshots: np.ndarray, radius: np.ndarray, size: int):
        centered = np.asarray(snapshots) - np.asarray(reference)[None, :]
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        return cls(np.asarray(reference), vh[:size].T, np.asarray(radius), singular_values)

    def project(self, wavefunctions: np.ndarray) -> np.ndarray:
        wavefunctions = np.atleast_2d(wavefunctions)
        weights = _trapezoid_sqrt_weights(self.radius)
        left = self.vectors * weights[:, None]
        right = (wavefunctions - self.reference[None, :]) * weights[None, :]
        coordinates, *_ = np.linalg.lstsq(left, right.T, rcond=None)
        return coordinates.T

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        return self.reference[None, :] + np.atleast_2d(coordinates) @ self.vectors.T


def _max_volume_indices(vectors: np.ndarray) -> np.ndarray:
    """Greedily select informative rows of a truncated potential SVD basis."""
    selected = []
    residual = vectors.copy()
    for _ in range(vectors.shape[1]):
        norms = np.linalg.norm(residual, axis=1)
        if selected:
            norms[selected] = -np.inf
        selected.append(int(np.argmax(norms)))
        chosen = vectors[selected]
        coefficients, *_ = np.linalg.lstsq(chosen.T, vectors.T, rcond=None)
        residual = vectors - (chosen.T @ coefficients).T
    return np.asarray(selected)


@dataclass
class LearnedROM:
    """Learn ``(I + sum p_j M_j)a = sum p_j b_j`` from FOM snapshots."""

    basis: ReducedBasis
    selected_indices: np.ndarray
    potential_reference: np.ndarray
    feature_scales: np.ndarray
    matrices: np.ndarray
    vectors: np.ndarray
    fit_residual: float

    @classmethod
    def fit(
        cls,
        reference_wavefunction: np.ndarray,
        training_wavefunctions: np.ndarray,
        radius: np.ndarray,
        reference_potential: np.ndarray,
        training_potentials: np.ndarray,
        basis_size: int,
        predictor_count: int,
    ):
        basis = ReducedBasis.fit(reference_wavefunction, training_wavefunctions, radius, basis_size)
        coordinates = basis.project(training_wavefunctions)

        delta_potential = (training_potentials - reference_potential[None, :]).T
        potential_modes, _, _ = np.linalg.svd(delta_potential, full_matrices=False)
        selected = _max_volume_indices(potential_modes[:, :predictor_count])
        raw_features = training_potentials[:, selected] - reference_potential[selected]
        scales = np.maximum(np.std(raw_features, axis=0), 1.0e-12)
        features = raw_features / scales

        sample_count, reduced_size = coordinates.shape
        block = reduced_size**2 + reduced_size
        design = np.zeros((sample_count * reduced_size, predictor_count * block), complex)
        target = -coordinates.reshape(-1)
        for n, (p, a) in enumerate(zip(features, coordinates)):
            for row in range(reduced_size):
                equation = n * reduced_size + row
                for j, feature in enumerate(p):
                    start = j * block
                    design[equation, start + row * reduced_size:start + (row + 1) * reduced_size] = feature * a
                    design[equation, start + reduced_size**2 + row] = -feature

        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        matrices = np.empty((predictor_count, reduced_size, reduced_size), complex)
        vectors = np.empty((predictor_count, reduced_size), complex)
        for j in range(predictor_count):
            start = j * block
            matrices[j] = solution[start:start + reduced_size**2].reshape(reduced_size, reduced_size)
            vectors[j] = solution[start + reduced_size**2:start + block]
        residual = float(np.mean(np.abs(design @ solution - target) ** 2))
        return cls(basis, selected, reference_potential, scales, matrices, vectors, residual)

    @property
    def selected_radii(self) -> np.ndarray:
        return self.basis.radius[self.selected_indices]

    def features(self, potentials: np.ndarray) -> np.ndarray:
        potentials = np.atleast_2d(potentials)
        return (potentials[:, self.selected_indices] - self.potential_reference[self.selected_indices]) / self.feature_scales

    def coordinates(self, potentials: np.ndarray) -> np.ndarray:
        features = self.features(potentials)
        identity = np.eye(self.basis.vectors.shape[1], dtype=complex)
        result = np.empty((len(features), len(identity)), complex)
        for n, p in enumerate(features):
            matrix = identity + np.einsum("j,jab->ab", p, self.matrices)
            rhs = np.einsum("j,ja->a", p, self.vectors)
            result[n] = np.linalg.solve(matrix, rhs)
        return result

    def predict(self, potentials: np.ndarray) -> np.ndarray:
        return self.basis.reconstruct(self.coordinates(potentials))
