"""Rectangular spatial discretizations for bounded ADR problems."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._numeric import real_array, real_scalar
from .backends.bounded import BoundedResidual


@dataclass(frozen=True)
class RectangleUpwind:
    """Uniform rectangular grid with centered diffusion and upwind advection."""

    bounds: tuple
    interior_points: tuple
    method = "rectangle-upwind"
    version = 1

    def assemble(self, model, *, time_steps):
        """Build the nodal order, quadrature, transport operators and point load."""
        return BoundedResidual(
            model, domain=self.bounds, interior_points=self.interior_points, time_steps=time_steps
        )

    def neumann_transfer_to(self, other):
        """Map concentrations between Neumann nodal control volumes of the same rectangle.

        The row weights are overlap measures divided by destination cell area.
        They preserve constants, nonnegativity and the spatial quadrature integral.
        Arrays use x-fastest ordering. The transpose transports Euclidean covectors.
        """
        if self.bounds != other.bounds:
            raise ValueError("Conservative transfer requires identical rectangular bounds")

        # The two outer control volumes have half width. Overlap fractions
        # are scale invariant, so each axis can be represented on [0, 1].
        def overlap(source, target):
            def edges(interior):
                intervals = interior + 1
                return np.r_[0.0, (np.arange(intervals) + 0.5) / intervals, 1.0]

            old, new = edges(source), edges(target)
            rows, columns, weights = [], [], []
            col = row = 0
            while row + 1 < len(new) and col + 1 < len(old):
                left = max(old[col], new[row])
                right = min(old[col + 1], new[row + 1])
                rows.append(row)
                columns.append(col)
                weights.append((right - left) / (new[row + 1] - new[row]))
                row, col = row + (new[row + 1] == right), col + (old[col + 1] == right)
            return sparse.csr_matrix(
                (weights, (rows, columns)), shape=(len(new) - 1, len(old) - 1)
            )

        x, y = (
            overlap(a, b) for a, b in zip(self.interior_points, other.interior_points, strict=True)
        )
        return sparse.kron(y, x, format="csr")


def spatial_discretization(method, bounds, spacing, *, coordinate_scale=1.0):
    """Partition physical bounds at the requested spacing and scale node coordinates."""
    if method != RectangleUpwind.method:
        raise ValueError(f"Unknown spatial discretization {method!r}")
    bounds = real_array(bounds, name="rectangular bounds")
    if bounds.shape != (4,):
        raise ValueError("Expected four finite rectangular bounds")
    spacing = real_scalar(spacing, name="spacing")
    coordinate_scale = real_scalar(coordinate_scale, name="coordinate_scale")
    if min(spacing, coordinate_scale) <= 0:
        raise ValueError("Grid spacing and coordinate scale must be positive finite scalars")
    intervals = np.diff(bounds.reshape(2, 2), axis=1).ravel() / spacing
    if (
        not np.isfinite(intervals).all()
        or np.any(intervals < 4)
        or not np.allclose(intervals, np.rint(intervals), rtol=0, atol=1e-10)
    ):
        raise ValueError("The spacing must partition both sides into at least four intervals")
    return RectangleUpwind(
        tuple((bounds / coordinate_scale).tolist()),
        tuple((np.rint(intervals).astype(int) - 1).tolist()),
    )
