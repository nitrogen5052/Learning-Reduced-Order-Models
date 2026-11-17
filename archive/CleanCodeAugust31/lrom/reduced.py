"""Centered reduced bases and the learned implicit reduced equation."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

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


def _trapezoid_sqrt_weights(coordinate: np.ndarray) -> np.ndarray:
    """Return square-root trapezoid weights for a one-dimensional mesh."""
    widths = np.diff(coordinate)
    weights = np.empty_like(coordinate)
    weights[0], weights[-1] = widths[0] / 2, widths[-1] / 2
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2
    return np.sqrt(weights)


@dataclass
class ReducedBasis:
    """A central reference wavefunction plus SVD modes of its deviations."""

    reference: np.ndarray
    vectors: np.ndarray
    coordinate: np.ndarray
    singular_values: np.ndarray

    @classmethod
    def fit(cls, reference: np.ndarray, snapshots: np.ndarray, coordinate: np.ndarray, size: int):
        """Build ``size`` SVD modes from snapshots centered on ``reference``."""
        centered = np.asarray(snapshots) - np.asarray(reference)[None, :]
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        return cls(np.asarray(reference), vh[:size].T, np.asarray(coordinate), singular_values)

    @property
    def radius(self) -> np.ndarray:
        """Deprecated alias for :attr:`coordinate` (which is normally ``s``)."""
        return self.coordinate

    def project(self, wavefunctions: np.ndarray) -> np.ndarray:
        """Return weighted least-squares coordinates for input wavefunctions."""
        wavefunctions = np.atleast_2d(wavefunctions)
        weights = _trapezoid_sqrt_weights(self.coordinate)
        left = self.vectors * weights[:, None]
        right = (wavefunctions - self.reference[None, :]) * weights[None, :]
        coordinates, *_ = np.linalg.lstsq(left, right.T, rcond=None)
        return coordinates.T

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        """Reconstruct wavefunctions from rows of reduced coordinates."""
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
    potential_predictor_count: int
    potential_reference: np.ndarray
    feature_scales: np.ndarray
    auxiliary_reference: np.ndarray
    matrices: np.ndarray
    vectors: np.ndarray
    fit_residual: float
    ridge: float

    @classmethod
    def fit(
        cls,
        reference_wavefunction: np.ndarray,
        training_wavefunctions: np.ndarray,
        coordinate: np.ndarray,
        reference_potential: np.ndarray,
        training_potentials: np.ndarray,
        basis_size: int,
        predictor_count: int,
        minimum_predictor_radius: float | None = None,
        additional_features: np.ndarray | None = None,
        reference_additional_features: np.ndarray | None = None,
        precomputed_basis: ReducedBasis | None = None,
        preselected_indices: np.ndarray | None = None,
        ridge: float = 0.0,
    ):
        """Fit a reduced basis and learned implicit coordinate equation.

        The wavefunction and potential arrays contain one row per training
        case and one column per radial mesh point. The returned model predicts
        reduced coordinates and reconstructed wavefunctions from new
        potential rows.
        """
        basis = (
            ReducedBasis.fit(
                reference_wavefunction, training_wavefunctions, coordinate, basis_size
            )
            if precomputed_basis is None else precomputed_basis
        )
        if basis.vectors.shape != (len(coordinate), basis_size):
            raise ValueError("precomputed basis does not match coordinate or basis size")
        coordinates = basis.project(training_wavefunctions)

        delta_potential = (training_potentials - reference_potential[None, :]).T
        if minimum_predictor_radius is None:
            eligible = np.arange(len(coordinate))
        else:
            eligible = np.flatnonzero(np.asarray(coordinate) >= minimum_predictor_radius)
            if eligible.size < predictor_count:
                raise ValueError(
                    "minimum_predictor_radius leaves fewer eligible mesh points "
                    "than requested predictors"
                )

        # Select from the requested physical region, then map the local row
        # indices back to the full radial mesh used by the FOM and the basis.
        if preselected_indices is None:
            delta_potential = delta_potential[eligible]
            potential_modes, _, _ = np.linalg.svd(
                delta_potential, full_matrices=False
            )
            selected = eligible[
                _max_volume_indices(potential_modes[:, :predictor_count])
            ]
        else:
            selected = np.asarray(preselected_indices, dtype=int).reshape(-1)
            if len(selected) != predictor_count:
                raise ValueError("preselected_indices has the wrong length")
            if (minimum_predictor_radius is not None and np.any(
                    np.asarray(coordinate)[selected] < minimum_predictor_radius)):
                raise ValueError("preselected predictor lies below the minimum coordinate")
        raw_potential_features = (
            training_potentials[:, selected] - reference_potential[selected]
        )
        if additional_features is None:
            auxiliary = np.empty((len(training_potentials), 0), dtype=float)
            auxiliary_reference = np.empty(0, dtype=float)
        else:
            auxiliary = np.atleast_2d(np.asarray(additional_features, dtype=float))
            if auxiliary.shape[0] != len(training_potentials):
                raise ValueError("additional features need one row per training case")
            auxiliary_reference = np.asarray(
                reference_additional_features, dtype=float
            ).reshape(-1)
            if auxiliary.shape[1] != len(auxiliary_reference):
                raise ValueError("reference additional features have the wrong size")
            auxiliary = auxiliary - auxiliary_reference[None, :]
        raw_features = np.column_stack([raw_potential_features, auxiliary])
        scales = np.maximum(np.std(raw_features, axis=0), 1.0e-12)
        features = raw_features / scales

        sample_count, reduced_size = coordinates.shape
        total_predictors = features.shape[1]
        coefficient_count = total_predictors * (reduced_size + 1)
        sampling_ratio = sample_count / coefficient_count

        # Each row of the learned implicit equation is a separate linear
        # regression with N_features * (N_basis + 1) complex coefficients.
        # Warn before least squares silently returns a fragile or non-unique
        # solution.  Auxiliary controls such as E, A, and Z are included in
        # total_predictors, so the count reflects the complete learned model.
        if sample_count < coefficient_count:
            warnings.warn(
                "Learned implicit fit is underdetermined: "
                f"{sample_count} training samples for {coefficient_count} "
                "complex coefficients per equation row "
                f"({total_predictors} features x "
                f"({reduced_size} basis coordinates + 1)). "
                "Increase the number of training samples, reduce the basis "
                "or predictor count, or introduce an explicitly validated "
                "regularization strategy.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif sampling_ratio < 2.0:
            warnings.warn(
                "Learned implicit fit has a small sampling margin: "
                f"{sample_count} training samples for {coefficient_count} "
                "complex coefficients per equation row "
                f"(sample/parameter ratio = {sampling_ratio:.2f}). "
                "Independent testing is strongly recommended.",
                UserWarning,
                stacklevel=2,
            )

        # Each output row of the implicit equation contains a different row of
        # M_j and b_j but the same feature/coordinate design.  Solving those
        # independent equations as multiple right-hand sides is exactly
        # equivalent to the former block-diagonal system, while reducing a
        # (samples*n) x (features*(n*n+n)) regression to
        # samples x (features*(n+1)).
        matrix_terms = np.einsum(
            "sf,si->sfi", features, coordinates, optimize=True
        )
        design = np.concatenate(
            [matrix_terms, -features[..., None]], axis=2
        ).reshape(sample_count, total_predictors * (reduced_size + 1))
        target = -coordinates
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if ridge == 0.0:
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        else:
            # Dimensionless ridge strength: alpha is measured relative to the
            # largest eigenvalue of D^H D, so a value such as 1e-14 retains a
            # comparable meaning when training data or feature scales change.
            left, singular_values, right_h = np.linalg.svd(
                design, full_matrices=False
            )
            alpha = ridge * singular_values[0] ** 2
            filtered = singular_values / (singular_values**2 + alpha)
            solution = (
                right_h.conj().T
                @ (filtered[:, None] * (left.conj().T @ target))
            )
        blocks = solution.reshape(
            total_predictors, reduced_size + 1, reduced_size
        )
        matrices = np.transpose(blocks[:, :reduced_size, :], (0, 2, 1))
        vectors = blocks[:, reduced_size, :]
        residual = (
            float(np.mean(np.abs(design @ solution - target) ** 2))
            if reduced_size else 0.0
        )
        return cls(
            basis, selected, predictor_count, reference_potential, scales,
            auxiliary_reference, matrices, vectors, residual, float(ridge),
        )

    @property
    def selected_radii(self) -> np.ndarray:
        """Deprecated alias for selected coordinates (normally values of ``s``)."""
        return self.selected_coordinates

    @property
    def selected_coordinates(self) -> np.ndarray:
        """Return the scaled coordinates selected as operator predictors."""
        return self.basis.coordinate[self.selected_indices]

    def features(self, potentials: np.ndarray) -> np.ndarray:
        """Convert potential rows into normalized selected-point features."""
        potentials = np.atleast_2d(potentials)
        return self.features_from_selected_values(potentials[:, self.selected_indices])

    def features_from_selected_values(
        self,
        values: np.ndarray,
        additional_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize potential values already evaluated at predictor radii."""
        values = np.atleast_2d(values)
        if values.shape[1] != len(self.selected_indices):
            raise ValueError("one value is required at every selected predictor radius")
        reference = self.potential_reference[self.selected_indices]
        raw = values - reference
        if len(self.auxiliary_reference):
            if additional_features is None:
                raise ValueError("this model also requires its additional features")
            auxiliary = np.atleast_2d(np.asarray(additional_features, dtype=float))
            if auxiliary.shape != (len(values), len(self.auxiliary_reference)):
                raise ValueError("additional features have the wrong shape")
            raw = np.column_stack([
                raw, auxiliary - self.auxiliary_reference[None, :]
            ])
        return raw / self.feature_scales

    def coordinates_from_features(self, features: np.ndarray) -> np.ndarray:
        """Solve the learned implicit equation from normalized features."""
        features = np.atleast_2d(features)
        if features.shape[1] != len(self.matrices):
            raise ValueError("feature count does not match the learned predictors")
        identity = np.eye(self.basis.vectors.shape[1], dtype=complex)
        result = np.empty((len(features), len(identity)), complex)
        for n, p in enumerate(features):
            matrix = identity + np.einsum("j,jab->ab", p, self.matrices)
            rhs = np.einsum("j,ja->a", p, self.vectors)
            result[n] = np.linalg.solve(matrix, rhs)
        return result

    def coordinates(self, potentials: np.ndarray) -> np.ndarray:
        """Solve the learned implicit system for each input potential row."""
        return self.coordinates_from_features(self.features(potentials))

    def predict(self, potentials: np.ndarray) -> np.ndarray:
        """Return reconstructed wavefunctions for input potential rows."""
        return self.basis.reconstruct(self.coordinates(potentials))
