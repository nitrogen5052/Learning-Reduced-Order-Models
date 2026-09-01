"""User-facing learned emulator and reusable prediction objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .channels import Channel
from .data import TrainingData
from .fom import matching_index, scaled_derivative_at, s_matrix_from_scaled_boundary
from .observables import ElasticCrossSectionKernel, assemble_elastic_cross_sections
from .problem import ScatteringProblem
from .reduced import LearnedROM, ReducedBasis


def _control_values(
    problem: ScatteringProblem,
    sample: Mapping[str, float | int],
    names: tuple[str, ...],
) -> np.ndarray:
    """Return scalar controls, including derived values supplied by a sample."""
    missing = [
        name for name in names
        if name not in sample and name not in problem.reference_sample
    ]
    if missing:
        raise KeyError(f"sample is missing control inputs: {missing}")
    return np.asarray([
        sample[name] if name in sample else problem.reference_sample[name]
        for name in names
    ], dtype=float)


@dataclass(frozen=True)
class BasisConfig:
    """Choose a fixed basis size or a wavefunction-compression tolerance."""

    size: int | None = None
    relative_svd_tolerance: float | None = None
    relative_l2_tolerance: float | None = None
    max_size: int | None = None
    allowed_sizes: tuple[int, ...] | None = None
    snapshot_scaling: str = "none"

    def __post_init__(self) -> None:
        """Require exactly one unambiguous basis-selection rule."""
        choices = (
            self.size is not None,
            self.relative_svd_tolerance is not None,
            self.relative_l2_tolerance is not None,
        )
        if sum(choices) != 1:
            raise ValueError(
                "specify exactly one of size, relative_svd_tolerance, or "
                "relative_l2_tolerance"
            )
        for value in (self.relative_svd_tolerance, self.relative_l2_tolerance):
            if value is not None and not 0 < float(value) < 1:
                raise ValueError("relative tolerances must lie between zero and one")

    def scale(
        self, snapshots: np.ndarray, coordinate: np.ndarray
    ) -> np.ndarray:
        """Apply the requested snapshot convention without modifying input data.

        ``"rose"`` reproduces ROSE's optional scaling ``phi / integral|phi|^2``.
        It is useful when one basis spans energies or nuclei because arbitrary
        shooting amplitudes otherwise dominate the SVD. ``"none"`` preserves
        the raw positive-derivative shooting normalization.
        """
        values = np.asarray(snapshots, dtype=complex).copy()
        if self.snapshot_scaling == "none":
            return values
        if self.snapshot_scaling == "rose":
            rows = np.atleast_2d(values)
            norms = np.trapezoid(np.abs(rows) ** 2, coordinate, axis=1)
            return rows / norms[:, None]
        raise ValueError("snapshot_scaling must be 'none' or 'rose'")

    def choose_size(self, snapshots: np.ndarray, reference: np.ndarray) -> int:
        """Return the reduced size implied by this configuration."""
        available = min(np.asarray(snapshots).shape)
        if self.size is not None:
            size = int(self.size)
        elif (self.relative_svd_tolerance is not None
              or self.relative_l2_tolerance is not None):
            _, singular, vh = np.linalg.svd(
                np.asarray(snapshots) - np.asarray(reference)[None, :],
                full_matrices=False,
            )
            if self.relative_svd_tolerance is not None:
                return self.choose_size_from_singular_values(singular)
            return self.choose_size_from_reconstruction(
                snapshots, reference, np.arange(np.asarray(snapshots).shape[1]), vh
            )
        else:
            raise ValueError("specify size or relative_svd_tolerance")
        return self._constrain_size(size, available)

    def choose_size_from_reconstruction(
        self,
        snapshots: np.ndarray,
        reference: np.ndarray,
        coordinate: np.ndarray,
        right_singular_vectors: np.ndarray,
    ) -> int:
        """Choose the smallest size meeting every relative-L2 training error.

        Size zero is tested explicitly and represents the unchanged reference
        wavefunction.  The SVD vectors define nested trial spaces, while the
        error and projections use trapezoid-weighted L2 norms.
        """
        if self.relative_l2_tolerance is None:
            raise ValueError("relative_l2_tolerance is not configured")
        snapshots = np.atleast_2d(np.asarray(snapshots, dtype=complex))
        reference = np.asarray(reference, dtype=complex).reshape(-1)
        coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
        vectors = np.asarray(right_singular_vectors, dtype=complex).T
        available = min(vectors.shape[1], snapshots.shape[0], snapshots.shape[1])
        if self.max_size is not None:
            available = min(available, int(self.max_size))

        widths = np.diff(coordinate)
        weights = np.empty_like(coordinate)
        weights[0], weights[-1] = widths[0] / 2, widths[-1] / 2
        weights[1:-1] = (widths[:-1] + widths[1:]) / 2
        centered = snapshots - reference[None, :]
        centered_norm2 = np.sum(np.abs(centered) ** 2 * weights[None, :], axis=1)
        snapshot_norm2 = np.maximum(
            np.sum(np.abs(snapshots) ** 2 * weights[None, :], axis=1), 1.0e-300
        )
        tolerance2 = float(self.relative_l2_tolerance) ** 2
        if np.max(centered_norm2 / snapshot_norm2) <= tolerance2:
            return self._constrain_size(0, max(available, 0))

        weighted_vectors = vectors[:, :available] * np.sqrt(weights)[:, None]
        weighted_centered = centered * np.sqrt(weights)[None, :]
        cross = weighted_vectors.conj().T @ weighted_centered.T
        selected_size = available
        for size in range(1, available + 1):
            left = weighted_vectors[:, :size]
            gram = left.conj().T @ left
            coefficients = np.linalg.solve(gram, cross[:size])
            captured = np.real(np.sum(coefficients.conj() * cross[:size], axis=0))
            residual2 = np.maximum(centered_norm2 - captured, 0.0)
            if np.max(residual2 / snapshot_norm2) <= tolerance2:
                selected_size = size
                break
        return self._constrain_size(selected_size, available)

    def choose_size_from_singular_values(self, singular_values: np.ndarray) -> int:
        """Choose a size from a decomposition already needed by the basis."""
        singular = np.asarray(singular_values, dtype=float).reshape(-1)
        if not len(singular):
            raise ValueError("singular_values cannot be empty")
        if self.size is not None:
            size = int(self.size)
        elif self.relative_svd_tolerance is not None:
            total = np.sum(singular**2)
            if total == 0:
                size = 1
            else:
                discarded = np.sqrt(
                    np.maximum(0.0, total - np.cumsum(singular**2)) / total
                )
                acceptable = np.flatnonzero(discarded <= self.relative_svd_tolerance)
                size = int(acceptable[0] + 1) if len(acceptable) else len(singular)
        else:
            raise ValueError("specify size or relative_svd_tolerance")
        return self._constrain_size(size, len(singular))

    def _constrain_size(self, size: int, available: int) -> int:
        """Apply caps and optional packing buckets to one raw size."""
        if self.max_size is not None:
            size = min(size, int(self.max_size))
        allow_zero = self.relative_l2_tolerance is not None
        if self.allowed_sizes is not None:
            allowed = tuple(sorted({int(value) for value in self.allowed_sizes}))
            if not allowed or allowed[0] < (0 if allow_zero else 1):
                raise ValueError("allowed basis sizes are outside the permitted range")
            candidates = [value for value in allowed if value >= size]
            if not candidates:
                raise ValueError("allowed basis sizes do not reach the requested size")
            size = candidates[0]
        minimum = 0 if allow_zero else 1
        if not minimum <= size <= available:
            raise ValueError(
                f"basis size must lie between {minimum} and {available}"
            )
        return size


@dataclass(frozen=True)
class MaxVol:
    """Configuration for max-volume predictor selection on the scaled mesh."""

    count: int
    minimum_s: float = 0.1
    ridge: float = 0.0

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("predictor count must be positive")
        if self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative")


@dataclass
class _ChannelModel:
    """One channel model with precomputed value/derivative boundary maps."""

    learned: LearnedROM
    ell: int
    spin: float
    matching_s: float
    boundary_reference: complex
    boundary_vectors: np.ndarray
    derivative_reference: complex
    derivative_vectors: np.ndarray
    control_inputs: tuple[str, ...]

    def features(
        self, problem: ScatteringProblem, sample: Mapping[str, float | int]
    ) -> np.ndarray:
        """Return normalized potential and optional scalar control features."""
        values = problem.scaled_potential(
            sample, self.spin, self.learned.selected_coordinates
        )
        controls = _control_values(problem, sample, self.control_inputs)
        return self.learned.features_from_selected_values(
            values, controls if len(controls) else None
        )[0]

    def coordinates(
        self, problem: ScatteringProblem, sample: Mapping[str, float | int]
    ) -> np.ndarray:
        """Predict reduced coordinates using only selected online features."""
        return self.learned.coordinates_from_features(
            self.features(problem, sample)
        )[0]

    def boundary_values(self, coordinates: np.ndarray) -> tuple[complex, complex]:
        """Return ``phi(s0)`` and ``dphi/ds(s0)`` from reduced coordinates."""
        value = self.boundary_reference + self.boundary_vectors @ coordinates
        derivative = self.derivative_reference + self.derivative_vectors @ coordinates
        return value, derivative

    def wavefunction(self, coordinates: np.ndarray) -> np.ndarray:
        """Explicitly reconstruct the full radial wavefunction when requested."""
        return self.learned.basis.reconstruct(coordinates)[0]


@dataclass
class _PackedOnlineModel:
    """Contiguous channel data used by the fast observable-only path."""

    channels: tuple[Channel, ...]
    spins: np.ndarray
    selected_s: np.ndarray
    selected_reference: np.ndarray
    feature_scales: np.ndarray
    auxiliary_reference: np.ndarray
    matrices: np.ndarray
    vectors: np.ndarray
    boundary_reference: np.ndarray
    boundary_vectors: np.ndarray
    derivative_reference: np.ndarray
    derivative_vectors: np.ndarray
    h_minus: np.ndarray
    h_plus: np.ndarray
    dh_minus: np.ndarray
    dh_plus: np.ndarray
    plus_indices: np.ndarray
    minus_indices: np.ndarray

    @classmethod
    def build(
        cls,
        problem: ScatteringProblem,
        models: Mapping[Channel, _ChannelModel],
        channels: tuple[Channel, ...] | None = None,
    ) -> "_PackedOnlineModel":
        """Pack equally sized channel models and cache matching functions.

        ``channels`` may select one equal-shape block from a larger ragged
        collection. Observable indices are populated only for the complete
        problem channel ordering.
        """
        channels = tuple(problem.channels) if channels is None else tuple(channels)
        ordered = [models[channel] for channel in channels]
        shapes = {
            (model.learned.matrices.shape, model.learned.vectors.shape)
            for model in ordered
        }
        if len(shapes) != 1:
            raise ValueError(
                "the fast cross-section route requires equal reduced sizes "
                "and predictor counts in every channel"
            )
        ells = np.asarray([channel[0] for channel in channels], dtype=int)
        spins = np.asarray([channel[1] for channel in channels], dtype=float)
        matching_s = np.asarray([model.matching_s for model in ordered])
        j = spherical_jn(ells, matching_s)
        y = spherical_yn(ells, matching_s)
        jp = spherical_jn(ells, matching_s, derivative=True)
        yp = spherical_yn(ells, matching_s, derivative=True)
        h_minus = matching_s * (j - 1j * y)
        h_plus = matching_s * (j + 1j * y)
        dh_minus = (j - 1j * y) + matching_s * (jp - 1j * yp)
        dh_plus = (j + 1j * y) + matching_s * (jp + 1j * yp)
        plus: list[int] = []
        minus: list[int] = []
        if channels == tuple(problem.channels):
            lookup = {channel: index for index, channel in enumerate(channels)}
            plus = [lookup[(0, 0.0)]]
            minus = [lookup[(0, 0.0)]]
            for ell in range(1, problem.l_max + 1):
                plus.append(lookup[(ell, float(ell))])
                minus.append(lookup[(ell, float(-(ell + 1)))])
        return cls(
            channels=channels,
            spins=spins,
            selected_s=np.asarray([
                model.learned.selected_coordinates for model in ordered
            ]),
            selected_reference=np.asarray([
                model.learned.potential_reference[model.learned.selected_indices]
                for model in ordered
            ]),
            feature_scales=np.asarray([
                model.learned.feature_scales for model in ordered
            ]),
            auxiliary_reference=np.asarray([
                model.learned.auxiliary_reference for model in ordered
            ]),
            matrices=np.asarray([model.learned.matrices for model in ordered]),
            vectors=np.asarray([model.learned.vectors for model in ordered]),
            boundary_reference=np.asarray([
                model.boundary_reference for model in ordered
            ]),
            boundary_vectors=np.asarray([
                model.boundary_vectors for model in ordered
            ]),
            derivative_reference=np.asarray([
                model.derivative_reference for model in ordered
            ]),
            derivative_vectors=np.asarray([
                model.derivative_vectors for model in ordered
            ]),
            h_minus=h_minus,
            h_plus=h_plus,
            dh_minus=dh_minus,
            dh_plus=dh_plus,
            plus_indices=np.asarray(plus),
            minus_indices=np.asarray(minus),
        )

    @classmethod
    def build_padded(
        cls,
        problem: ScatteringProblem,
        models: Mapping[Channel, _ChannelModel],
    ) -> "_PackedOnlineModel":
        """Pad variable channel bases with exact zero coordinates.

        Every channel is embedded in the largest active reduced space.  The
        inactive rows retain the identity from ``I + sum(p_j M_j)`` and have
        zero right-hand sides, so their coordinates are exactly zero.  This
        preserves the smaller learned models while enabling one batched solve.
        """
        channels = tuple(problem.channels)
        ordered = [models[channel] for channel in channels]
        feature_counts = {model.learned.matrices.shape[0] for model in ordered}
        predictor_counts = {
            len(model.learned.selected_coordinates) for model in ordered
        }
        if len(feature_counts) != 1 or len(predictor_counts) != 1:
            raise ValueError(
                "zero-padded packing requires equal feature and predictor "
                "counts in every channel"
            )

        maximum_size = max(
            model.learned.basis.vectors.shape[1] for model in ordered
        )
        feature_count = next(iter(feature_counts))
        matrices = np.zeros(
            (len(channels), feature_count, maximum_size, maximum_size),
            dtype=complex,
        )
        vectors = np.zeros(
            (len(channels), feature_count, maximum_size), dtype=complex
        )
        boundary_vectors = np.zeros(
            (len(channels), maximum_size), dtype=complex
        )
        derivative_vectors = np.zeros_like(boundary_vectors)
        for index, model in enumerate(ordered):
            size = model.learned.basis.vectors.shape[1]
            matrices[index, :, :size, :size] = model.learned.matrices
            vectors[index, :, :size] = model.learned.vectors
            boundary_vectors[index, :size] = model.boundary_vectors
            derivative_vectors[index, :size] = model.derivative_vectors

        ells = np.asarray([channel[0] for channel in channels], dtype=int)
        spins = np.asarray([channel[1] for channel in channels], dtype=float)
        matching_s = np.asarray([model.matching_s for model in ordered])
        j = spherical_jn(ells, matching_s)
        y = spherical_yn(ells, matching_s)
        jp = spherical_jn(ells, matching_s, derivative=True)
        yp = spherical_yn(ells, matching_s, derivative=True)
        lookup = {channel: index for index, channel in enumerate(channels)}
        plus = [lookup[(0, 0.0)]]
        minus = [lookup[(0, 0.0)]]
        for ell in range(1, problem.l_max + 1):
            plus.append(lookup[(ell, float(ell))])
            minus.append(lookup[(ell, float(-(ell + 1)))])
        return cls(
            channels=channels,
            spins=spins,
            selected_s=np.asarray([
                model.learned.selected_coordinates for model in ordered
            ]),
            selected_reference=np.asarray([
                model.learned.potential_reference[model.learned.selected_indices]
                for model in ordered
            ]),
            feature_scales=np.asarray([
                model.learned.feature_scales for model in ordered
            ]),
            auxiliary_reference=np.asarray([
                model.learned.auxiliary_reference for model in ordered
            ]),
            matrices=matrices,
            vectors=vectors,
            boundary_reference=np.asarray([
                model.boundary_reference for model in ordered
            ]),
            boundary_vectors=boundary_vectors,
            derivative_reference=np.asarray([
                model.derivative_reference for model in ordered
            ]),
            derivative_vectors=derivative_vectors,
            h_minus=matching_s * (j - 1j * y),
            h_plus=matching_s * (j + 1j * y),
            dh_minus=(j - 1j * y) + matching_s * (jp - 1j * yp),
            dh_plus=(j + 1j * y) + matching_s * (jp + 1j * yp),
            plus_indices=np.asarray(plus),
            minus_indices=np.asarray(minus),
        )

    def s_matrices(
        self,
        problem: ScatteringProblem,
        sample: Mapping[str, float | int],
        control_inputs: tuple[str, ...],
    ) -> tuple[np.ndarray, float]:
        """Evaluate all channel predictors and reduced systems in one batch."""
        system = problem.system(sample)
        omega = problem.omega(sample)
        radii = self.selected_s / system.wave_number
        selected_values = problem.potential.evaluate(
            radii, omega, self.spins[:, None]
        ) / system.com_energy
        raw_features = selected_values - self.selected_reference
        if control_inputs:
            controls = _control_values(problem, sample, control_inputs)
            control_rows = controls[None, :] - self.auxiliary_reference
            raw_features = np.concatenate([raw_features, control_rows], axis=1)
        features = raw_features / self.feature_scales
        return self.s_matrices_from_features(features), system.wave_number

    def s_matrices_batch(
        self,
        problem: ScatteringProblem,
        samples: list[Mapping[str, float | int]],
        control_inputs: tuple[str, ...],
        linear_solver=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a sample batch while sharing channel contractions.

        Potential construction remains sample-local because each case has its
        own kinematics and optical parameters.  The expensive learned-equation
        contractions and solves are assembled once over the combined
        ``(sample, channel)`` stack, eliminating the Python loop around four
        small einsums and one angular-kernel evaluation per case.
        """
        feature_rows, wave_numbers = self.feature_batch(
            problem, samples, control_inputs
        )
        return self.s_matrices_from_feature_batch(
            feature_rows, linear_solver=linear_solver
        ), wave_numbers

    def feature_batch(
        self,
        problem: ScatteringProblem,
        samples: list[Mapping[str, float | int]],
        control_inputs: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build normalized online features without solving reduced systems."""
        if not samples:
            return (
                np.empty(
                    (0, len(self.channels), self.matrices.shape[1]),
                    dtype=complex,
                ),
                np.empty(0, dtype=float),
            )
        feature_rows = np.empty(
            (len(samples), len(self.channels), self.matrices.shape[1]),
            dtype=complex,
        )
        wave_numbers = np.empty(len(samples), dtype=float)
        for index, sample in enumerate(samples):
            system = problem.system(sample)
            wave_numbers[index] = system.wave_number
            omega = problem.omega(sample)
            radii = self.selected_s / system.wave_number
            selected_values = problem.potential.evaluate(
                radii, omega, self.spins[:, None]
            ) / system.com_energy
            raw_features = selected_values - self.selected_reference
            if control_inputs:
                controls = _control_values(problem, sample, control_inputs)
                control_rows = controls[None, :] - self.auxiliary_reference
                raw_features = np.concatenate(
                    [raw_features, control_rows], axis=1
                )
            feature_rows[index] = raw_features / self.feature_scales
        return feature_rows, wave_numbers

    def s_matrices_from_features(self, features: np.ndarray) -> np.ndarray:
        """Solve this packed block from already normalized feature rows."""
        features = np.asarray(features)
        size = self.matrices.shape[-1]
        if size:
            systems = np.eye(size, dtype=complex)[None, :, :] + np.einsum(
                "cf,cfij->cij", features, self.matrices, optimize=False
            )
            right_sides = np.einsum(
                "cf,cfi->ci", features, self.vectors, optimize=False
            )
            # The explicit final axis keeps NumPy 2.x from interpreting a stacked
            # collection of right-hand sides as a matrix shared by every channel.
            coordinates = np.linalg.solve(systems, right_sides[..., None])[..., 0]
        else:
            coordinates = np.empty((len(self.channels), 0), dtype=complex)
        values = self.boundary_reference + np.einsum(
            "ci,ci->c", self.boundary_vectors, coordinates, optimize=False
        )
        derivatives = self.derivative_reference + np.einsum(
            "ci,ci->c", self.derivative_vectors, coordinates, optimize=False
        )
        log_derivatives = derivatives / values
        s_matrices = -(
            self.dh_minus - log_derivatives * self.h_minus
        ) / (
            self.dh_plus - log_derivatives * self.h_plus
        )
        return s_matrices

    def s_matrices_from_feature_batch(
        self, features: np.ndarray, linear_solver=None
    ) -> np.ndarray:
        """Solve samples and channels in one stacked learned-equation batch."""
        features = np.asarray(features)
        if features.ndim != 3 or features.shape[1:] != self.matrices.shape[:2]:
            raise ValueError(
                "batched features must have shape "
                "(samples, channels, feature_count)"
            )
        sample_count = len(features)
        size = self.matrices.shape[-1]
        if size:
            identity = np.eye(size, dtype=complex)
            if sample_count == 1:
                # Avoid the contraction planner's fixed cost for one query.
                systems = identity[None, None, :, :] + np.einsum(
                    "bcf,cfij->bcij", features, self.matrices,
                    optimize=False,
                )
                right_sides = np.einsum(
                    "bcf,cfi->bci", features, self.vectors,
                    optimize=False,
                )
                system_batch = systems.reshape(-1, size, size)
                rhs_batch = right_sides.reshape(-1, size)
                if linear_solver is None:
                    coordinates = np.linalg.solve(
                        system_batch, rhs_batch[..., None]
                    )[..., 0]
                else:
                    coordinates = linear_solver(system_batch, rhs_batch)
                coordinates = coordinates.reshape(
                    sample_count, len(self.channels), size
                )
            else:
                # Channel-major matrix products produce contiguous solve blocks.
                # The superficially simpler optimized einsum returns a layout
                # that NumPy must copy twice before calling the batched solver.
                channel_features = features.transpose(1, 0, 2)
                channel_count = len(self.channels)
                systems = identity[None, None, :, :] + np.matmul(
                    channel_features,
                    self.matrices.reshape(
                        channel_count, self.matrices.shape[1], size * size
                    ),
                ).reshape(channel_count, sample_count, size, size)
                right_sides = np.matmul(channel_features, self.vectors)
                system_batch = systems.reshape(-1, size, size)
                rhs_batch = right_sides.reshape(-1, size)
                if linear_solver is None:
                    coordinates = np.linalg.solve(
                        system_batch, rhs_batch[..., None]
                    )[..., 0]
                else:
                    coordinates = linear_solver(system_batch, rhs_batch)
                coordinates = coordinates.reshape(
                    channel_count, sample_count, size
                ).transpose(1, 0, 2)
        else:
            coordinates = np.empty(
                (sample_count, len(self.channels), 0), dtype=complex
            )
        values = self.boundary_reference[None, :] + np.einsum(
            "ci,bci->bc", self.boundary_vectors, coordinates, optimize=False
        )
        derivatives = self.derivative_reference[None, :] + np.einsum(
            "ci,bci->bc", self.derivative_vectors, coordinates, optimize=False
        )
        log_derivatives = derivatives / values
        return -(
            self.dh_minus[None, :] - log_derivatives * self.h_minus[None, :]
        ) / (
            self.dh_plus[None, :] - log_derivatives * self.h_plus[None, :]
        )


@dataclass
class _RaggedPackedOnlineModel:
    """Equal-shape channel blocks for fast variable-basis evaluation."""

    channels: tuple[Channel, ...]
    blocks: tuple[_PackedOnlineModel, ...]
    block_indices: tuple[np.ndarray, ...]
    plus_indices: np.ndarray
    minus_indices: np.ndarray
    spins: np.ndarray | None
    selected_s: np.ndarray | None
    selected_reference: np.ndarray | None
    feature_scales: np.ndarray | None
    auxiliary_reference: np.ndarray | None

    @classmethod
    def build(
        cls, problem: ScatteringProblem, models: Mapping[Channel, _ChannelModel]
    ) -> "_RaggedPackedOnlineModel":
        """Group channels by learned-system shape and pack each group."""
        channels = tuple(problem.channels)
        grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[Channel]] = {}
        for channel in channels:
            model = models[channel]
            shape = (model.learned.matrices.shape, model.learned.vectors.shape)
            grouped.setdefault(shape, []).append(channel)

        channel_lookup = {channel: index for index, channel in enumerate(channels)}
        block_channels = tuple(tuple(group) for group in grouped.values())
        blocks = tuple(
            _PackedOnlineModel.build(problem, models, group)
            for group in block_channels
        )
        block_indices = tuple(
            np.asarray([channel_lookup[channel] for channel in group], dtype=int)
            for group in block_channels
        )
        plus = [channel_lookup[(0, 0.0)]]
        minus = [channel_lookup[(0, 0.0)]]
        for ell in range(1, problem.l_max + 1):
            plus.append(channel_lookup[(ell, float(ell))])
            minus.append(channel_lookup[(ell, float(-(ell + 1)))])
        ordered = [models[channel] for channel in channels]
        predictor_counts = {
            len(model.learned.selected_coordinates) for model in ordered
        }
        if len(predictor_counts) == 1:
            spins = np.asarray([channel[1] for channel in channels], dtype=float)
            selected_s = np.asarray([
                model.learned.selected_coordinates for model in ordered
            ])
            selected_reference = np.asarray([
                model.learned.potential_reference[model.learned.selected_indices]
                for model in ordered
            ])
            feature_scales = np.asarray([
                model.learned.feature_scales for model in ordered
            ])
            auxiliary_reference = np.asarray([
                model.learned.auxiliary_reference for model in ordered
            ])
        else:
            spins = selected_s = selected_reference = None
            feature_scales = auxiliary_reference = None
        return cls(
            channels, blocks, block_indices,
            np.asarray(plus, dtype=int), np.asarray(minus, dtype=int),
            spins, selected_s, selected_reference,
            feature_scales, auxiliary_reference,
        )

    def s_matrices(
        self,
        problem: ScatteringProblem,
        sample: Mapping[str, float | int],
        control_inputs: tuple[str, ...],
    ) -> tuple[np.ndarray, float]:
        """Evaluate every equal-size block and restore channel order."""
        values = np.empty(len(self.channels), dtype=complex)
        system = problem.system(sample)
        wave_number = system.wave_number
        if self.selected_s is not None:
            omega = problem.omega(sample)
            radii = self.selected_s / wave_number
            selected_values = problem.potential.evaluate(
                radii, omega, self.spins[:, None]
            ) / system.com_energy
            raw_features = selected_values - self.selected_reference
            if control_inputs:
                controls = _control_values(problem, sample, control_inputs)
                raw_features = np.concatenate([
                    raw_features,
                    controls[None, :] - self.auxiliary_reference,
                ], axis=1)
            features = raw_features / self.feature_scales
            for block, indices in zip(self.blocks, self.block_indices):
                values[indices] = block.s_matrices_from_features(features[indices])
            return values, wave_number

        for block, indices in zip(self.blocks, self.block_indices):
            block_values, wave_number = block.s_matrices(
                problem, sample, control_inputs
            )
            values[indices] = block_values
        return values, wave_number


class Prediction:
    """Lazy, reusable LROM prediction for one named input selection."""

    def __init__(self, emulator: "ScatteringLROM", sample: Mapping[str, float | int]):
        self.emulator = emulator
        self.sample = dict(sample)
        self._coordinates: dict[Channel, np.ndarray] = {}
        self._s_matrices: dict[Channel, complex] = {}

    def coordinates(self, channel: Channel) -> np.ndarray:
        """Return and cache the reduced coordinates for one channel."""
        if channel not in self._coordinates:
            model = self.emulator.channel_models[channel]
            self._coordinates[channel] = model.coordinates(
                self.emulator.problem, self.sample
            )
        return self._coordinates[channel].copy()

    def s_matrix(self, channel: Channel) -> complex:
        """Return S from the reduced boundary value and derivative only."""
        if channel not in self._s_matrices:
            model = self.emulator.channel_models[channel]
            value, derivative = model.boundary_values(self.coordinates(channel))
            self._s_matrices[channel] = s_matrix_from_scaled_boundary(
                model.matching_s, value, derivative, model.ell
            )
        return self._s_matrices[channel]

    def condition_number(self, channel: Channel) -> float:
        """Return the condition number of one learned implicit system."""
        model = self.emulator.channel_models[channel]
        if model.learned.basis.vectors.shape[1] == 0:
            return 1.0
        features = model.features(self.emulator.problem, self.sample)
        identity = np.eye(model.learned.basis.vectors.shape[1], dtype=complex)
        matrix = identity + np.einsum("j,jab->ab", features, model.learned.matrices)
        return float(np.linalg.cond(matrix))

    def cross_section(self, angles: np.ndarray) -> np.ndarray:
        """Assemble the elastic differential cross section in mb/sr."""
        return self.emulator._fast_cross_section(self.sample, angles)

    def wavefunction(self, channel: Channel) -> np.ndarray:
        """Reconstruct the full wavefunction on the emulator's ``s`` mesh."""
        model = self.emulator.channel_models[channel]
        return model.wavefunction(self.coordinates(channel))

    def physical_wavefunction(self, channel: Channel) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(r, phi)`` by mapping the shared ``s`` mesh through this k."""
        system = self.emulator.problem.system(self.sample)
        return system.radius, self.wavefunction(channel)


class ScatteringLROM:
    """Train and evaluate learned reduced models for a scattering problem."""

    def __init__(
        self,
        problem: ScatteringProblem,
        basis: BasisConfig,
        predictors: MaxVol,
        control_inputs: tuple[str, ...] = (),
        packing_strategy: str = "grouped",
    ):
        self.problem = problem
        self.basis_config = basis
        self.predictor_config = predictors
        self.control_inputs = tuple(control_inputs)
        self.packing_strategy = self._validate_packing_strategy(packing_strategy)
        self.channel_models: dict[Channel, _ChannelModel] = {}
        self.training_data: TrainingData | None = None
        self._packed_online: _PackedOnlineModel | _RaggedPackedOnlineModel | None = None
        self._angular_kernels: dict[bytes, ElasticCrossSectionKernel] = {}

    def train(
        self,
        data: TrainingData,
        *,
        precomputed_bases: Mapping[Channel, ReducedBasis] | None = None,
    ) -> "ScatteringLROM":
        """Fit every channel from a reusable full-order training database.

        ``precomputed_bases`` may be supplied when comparing alternative
        learned equations on exactly the same reduced spaces. This avoids
        repeating wavefunction SVDs and isolates changes in predictor design.
        """
        if tuple(data.channels) != tuple(self.problem.channels):
            raise ValueError("training channels do not match the problem")
        if precomputed_bases is not None:
            missing_bases = set(data.channels).difference(precomputed_bases)
            if missing_bases:
                raise ValueError(
                    f"precomputed bases are missing channels: {sorted(missing_bases)}"
                )
        models = {}
        missing_controls = set(self.control_inputs).difference(data.reference_sample)
        if missing_controls:
            raise ValueError(f"unknown control inputs: {sorted(missing_controls)}")
        additional = (
            np.asarray([[sample[name] for name in self.control_inputs]
                        for sample in data.samples], dtype=float)
            if self.control_inputs else None
        )
        reference_additional = (
            np.asarray([data.reference_sample[name] for name in self.control_inputs], float)
            if self.control_inputs else None
        )
        wave_cache = getattr(data, "_wave_svd_cache", {})
        predictor_cache = getattr(data, "_predictor_index_cache", {})
        for channel in data.channels:
            training_waves = self.basis_config.scale(
                data.wavefunctions[channel], data.s_mesh
            )
            reference_wave = self.basis_config.scale(
                data.reference_wavefunctions[channel][None, :], data.s_mesh
            )[0]
            precomputed_basis = (
                None if precomputed_bases is None
                else precomputed_bases[channel]
            )
            if precomputed_basis is not None:
                basis_size = precomputed_basis.vectors.shape[1]
            elif (self.basis_config.relative_svd_tolerance is not None
                    or self.basis_config.relative_l2_tolerance is not None):
                centered = training_waves - reference_wave[None, :]
                requested_vectors = min(
                    centered.shape[0], centered.shape[1],
                    int(self.basis_config.max_size)
                    if self.basis_config.max_size is not None
                    else min(centered.shape),
                )
                cache_key = (channel, self.basis_config.snapshot_scaling)
                cached = wave_cache.get(cache_key)
                if cached is None or cached[1].shape[0] < requested_vectors:
                    _, singular_values, full_vh = np.linalg.svd(
                        centered, full_matrices=False
                    )
                    vh = full_vh[:requested_vectors]
                    wave_cache[cache_key] = (singular_values, vh)
                else:
                    singular_values, vh = cached
                if self.basis_config.relative_svd_tolerance is not None:
                    basis_size = self.basis_config.choose_size_from_singular_values(
                        singular_values
                    )
                else:
                    basis_size = self.basis_config.choose_size_from_reconstruction(
                        training_waves, reference_wave, data.s_mesh, vh
                    )
                precomputed_basis = ReducedBasis(
                    reference_wave,
                    vh[:basis_size].T,
                    np.asarray(data.s_mesh),
                    singular_values,
                )
            else:
                basis_size = self.basis_config.choose_size(
                    training_waves, reference_wave
                )
            predictor_key = (
                channel, self.predictor_config.count,
                float(self.predictor_config.minimum_s),
            )
            learned = LearnedROM.fit(
                reference_wave, training_waves,
                data.s_mesh, data.reference_potentials[channel],
                data.potentials[channel], basis_size,
                self.predictor_config.count,
                self.predictor_config.minimum_s,
                additional_features=additional,
                reference_additional_features=reference_additional,
                precomputed_basis=precomputed_basis,
                preselected_indices=predictor_cache.get(predictor_key),
                ridge=self.predictor_config.ridge,
            )
            predictor_cache[predictor_key] = learned.selected_indices.copy()
            match = matching_index(data.s_mesh, self.problem.matching_s)
            reference_derivative = scaled_derivative_at(
                data.s_mesh, learned.basis.reference, match
            )
            vector_derivatives = scaled_derivative_at(
                data.s_mesh, learned.basis.vectors, match
            )
            models[channel] = _ChannelModel(
                learned, channel[0], channel[1],
                float(data.s_mesh[match]),
                learned.basis.reference[match],
                learned.basis.vectors[match, :],
                reference_derivative,
                vector_derivatives,
                self.control_inputs,
            )
        self.channel_models = models
        data._wave_svd_cache = wave_cache
        data._predictor_index_cache = predictor_cache
        self.training_data = data
        model_shapes = {
            (model.learned.matrices.shape, model.learned.vectors.shape)
            for model in models.values()
        }
        self._build_online_model(model_shapes)
        self._angular_kernels.clear()
        return self

    @staticmethod
    def _validate_packing_strategy(strategy: str) -> str:
        """Validate the online layout used for variable channel bases."""
        if strategy not in {"grouped", "padded"}:
            raise ValueError("packing_strategy must be 'grouped' or 'padded'")
        return strategy

    def _build_online_model(self, model_shapes=None) -> None:
        """Pack trained channels using the selected online layout."""
        if model_shapes is None:
            model_shapes = {
                (model.learned.matrices.shape, model.learned.vectors.shape)
                for model in self.channel_models.values()
            }
        if len(model_shapes) == 1:
            self._packed_online = _PackedOnlineModel.build(
                self.problem, self.channel_models
            )
        elif self.packing_strategy == "padded":
            self._packed_online = _PackedOnlineModel.build_padded(
                self.problem, self.channel_models
            )
        else:
            self._packed_online = _RaggedPackedOnlineModel.build(
                self.problem, self.channel_models
            )

    def repack(self, strategy: str) -> "ScatteringLROM":
        """Switch the online layout without retraining the channel models."""
        if not self.channel_models:
            raise RuntimeError("train the emulator before repacking")
        self.packing_strategy = self._validate_packing_strategy(strategy)
        self._build_online_model()
        self._angular_kernels.clear()
        return self

    def _fast_cross_section(
        self, sample: Mapping[str, float | int], angles: np.ndarray
    ) -> np.ndarray:
        """Evaluate the packed predictor-to-cross-section online route."""
        if not self.channel_models:
            raise RuntimeError("train the emulator before prediction")
        if self._packed_online is None:
            prediction = Prediction(self, sample)
            s_by_channel = {
                channel: np.asarray([prediction.s_matrix(channel)])
                for channel in self.problem.channels
            }
            wave_number = self.problem.system(sample).wave_number
            return assemble_elastic_cross_sections(
                s_by_channel, angles, wave_number
            )[0]
        s_channels, wave_number = self._packed_online.s_matrices(
            self.problem, sample, self.control_inputs
        )
        angle_values = np.ascontiguousarray(np.asarray(angles, dtype=float).reshape(-1))
        key = angle_values.tobytes()
        kernel = self._angular_kernels.get(key)
        if kernel is None:
            kernel = ElasticCrossSectionKernel.build(
                angle_values, 1.0, self.problem.l_max
            )
            self._angular_kernels[key] = kernel
        s_plus = s_channels[self._packed_online.plus_indices]
        s_minus = s_channels[self._packed_online.minus_indices]
        return kernel.evaluate(s_plus, s_minus) / wave_number**2

    def predict(self, sample: Mapping[str, float | int]) -> Prediction:
        """Create a lazy prediction object for one input selection."""
        if not self.channel_models:
            raise RuntimeError("train the emulator before prediction")
        return Prediction(self, sample)

    def cross_section(
        self, sample: Mapping[str, float | int], angles: np.ndarray
    ) -> np.ndarray:
        """Return a cross section through the packed observable-only route."""
        return self._fast_cross_section(sample, angles)

    def cross_sections(
        self,
        samples: list[Mapping[str, float | int]],
        angles: np.ndarray,
        linear_solver=None,
    ) -> np.ndarray:
        """Return one cross section per input with an optional batch solver."""
        if isinstance(self._packed_online, _PackedOnlineModel):
            s_channels, wave_numbers = self._packed_online.s_matrices_batch(
                self.problem, samples, self.control_inputs,
                linear_solver=linear_solver,
            )
            angle_values = np.ascontiguousarray(
                np.asarray(angles, dtype=float).reshape(-1)
            )
            key = angle_values.tobytes()
            kernel = self._angular_kernels.get(key)
            if kernel is None:
                kernel = ElasticCrossSectionKernel.build(
                    angle_values, 1.0, self.problem.l_max
                )
                self._angular_kernels[key] = kernel
            s_plus = s_channels[:, self._packed_online.plus_indices]
            s_minus = s_channels[:, self._packed_online.minus_indices]
            return kernel.evaluate(s_plus, s_minus) / wave_numbers[:, None] ** 2
        return np.asarray([
            self._fast_cross_section(sample, angles) for sample in samples
        ])

    def matrices(self, channel: Channel | None = None):
        """Return learned implicit matrices and vectors for inspection."""
        def values(key: Channel):
            learned = self.channel_models[key].learned
            return {"matrices": learned.matrices.copy(),
                    "vectors": learned.vectors.copy(),
                    "fit_residual": learned.fit_residual}
        return values(channel) if channel is not None else {
            key: values(key) for key in self.channel_models
        }
