"""Trajectory geometry for sEEG leads (pure numpy).

A :class:`SEEGLead` is a straight entry->tip trajectory; it computes the world-mm contact
centres from the catalog geometry. :func:`perp_distance_and_axial` is the shared
point-to-axis primitive used by the rasteriser and the field sampler. No SimNIBS import
here, so the geometry can be unit-tested standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import ElectrodeSpec


@dataclass(frozen=True)
class SEEGLead:
    """One depth lead: a straight trajectory from scalp ``entry`` to deep ``tip``.

    ``target_mm`` is the electrode TIP (deepest point). Contacts are numbered from
    the tip: contact 0 sits ``first_contact_depth_mm`` proximal to the tip and each
    subsequent contact is one ``spacing`` further toward the entry, matching clinical
    sEEG numbering (distal contact = 1).
    """

    name: str
    part_number: str
    entry_mm: tuple[float, float, float]
    target_mm: tuple[float, float, float]
    first_contact_depth_mm: float = 2.0
    n_contacts: int | None = None

    # --- basic vectors ---
    @property
    def A(self) -> np.ndarray:  # entry
        return np.asarray(self.entry_mm, dtype=float)

    @property
    def B(self) -> np.ndarray:  # tip / target
        return np.asarray(self.target_mm, dtype=float)

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.B - self.A))

    def axis_unit(self) -> np.ndarray:
        """Entry -> tip unit vector."""
        d = self.B - self.A
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError(f"lead {self.name!r}: entry and target coincide")
        return d / n

    def n(self, spec: "ElectrodeSpec") -> int:
        return int(self.n_contacts) if self.n_contacts is not None else int(spec.n_contacts)

    def contact_centers(self, spec: "ElectrodeSpec") -> np.ndarray:
        """(n, 3) world-mm contact centres, marching from the tip toward the entry.

        Offsets are measured from the tip (``target_mm``): contact 0 at
        ``first_contact_depth_mm``, then one ``spacing`` per contact. Validated so
        the metal of the deepest contact does not extend past the tip and the
        shallowest does not extend past the entry.
        """
        half = 0.5 * spec.contact_len_mm
        if self.first_contact_depth_mm < half:
            raise ValueError(
                f"lead {self.name!r}: first_contact_depth_mm="
                f"{self.first_contact_depth_mm} places contact-0 metal past the tip "
                f"(need >= contact_len/2 = {half:.2f} mm)"
            )
        u = self.axis_unit()
        n = self.n(spec)
        offsets = self.first_contact_depth_mm + np.arange(n) * spec.spacing_mm
        centers = self.B[None, :] - u[None, :] * offsets[:, None]
        # proximal end of the shallowest contact must stay within the trajectory
        span = float(offsets[-1]) + half if n else 0.0
        if span > self.length_mm:
            raise ValueError(
                f"lead {self.name!r}: array span {span:.1f} mm exceeds insertion "
                f"length {self.length_mm:.1f} mm (reduce n_contacts / spacing / "
                f"first_contact_depth, or lengthen the trajectory)"
            )
        return centers


def perp_distance_and_axial(
    bar: np.ndarray, A: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Perpendicular distance of points ``bar`` to segment A->B and their axial
    coordinate (mm from A along the axis, clamped to the segment).

    Returns (dist, axial) each of shape (N,).
    """
    d = B - A
    L2 = float(d @ d)
    if L2 == 0:
        raise ValueError("degenerate axis segment (A == B)")
    t = np.clip((bar - A) @ d / L2, 0.0, 1.0)
    proj = A + t[:, None] * d
    dist = np.linalg.norm(bar - proj, axis=1)
    axial = t * np.sqrt(L2)
    return dist, axial
