"""Sample a solved field along / around each electrode for validation.

Extracts, per lead, the field magnitude (a) along a line parallel to the electrode
axis just outside the body and (b) on radial rays outward from each contact centre.
The radial decay profile is what tests the Missey-2026 SI Fig S7 signature (2-7.2x
enhancement decaying to background within ~400 um). The electrode BODY tets
(contact + insulating shaft) are excluded from the interpolation so readings are the
field in *tissue* (incl. the glial sheath), never inside the conductor/insulator.

All sample points for a lead are gathered into a single ``interpolate_scattered``
call (which deep-copies the mesh internally) so a whole-head result mesh is sampled
once per lead, not once per ray.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._params import SEEG_CONTACT, SEEG_SHAFT

if TYPE_CHECKING:  # pragma: no cover
    from simnibs.mesh_tools import mesh_io
    from .embed import SEEGPlacement
    from .catalog import ElectrodeCatalog

_COMMON_FIELDS = ("TImax", "TI_max", "TI_amplitude", "magnE", "normE", "E")


@dataclass
class AxisProfile:
    """Sampled field for one lead."""

    lead_name: str
    contact_centers: np.ndarray            # (n, 3)
    axial_mm: np.ndarray                   # (Na,) distance along axis from tip
    axial_values: np.ndarray               # (Na,)
    radial_mm: np.ndarray                  # (Nr,) distance from contact surface
    radial_values: np.ndarray              # (n, Nr) per-contact radial profile
    background: float                      # far-field median (reference level)

    def enhancement(self) -> np.ndarray:
        """Per-contact peak / background ratio (compare to Missey 2-7.2x)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmax(self.radial_values, axis=1) / self.background

    def decay_length_mm(self, frac: float = 1.2) -> np.ndarray:
        """Radial distance at which each contact's profile falls to ``frac`` x
        background (compare to Missey ~0.4 mm). NaN if it never does."""
        out = np.full(self.radial_values.shape[0], np.nan)
        thr = frac * self.background
        for i, prof in enumerate(self.radial_values):
            below = np.where(prof <= thr)[0]
            if below.size:
                out[i] = self.radial_mm[below[0]]
        return out


def _perp_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to axis ``u``."""
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def _pick_field(result_mesh: "mesh_io.Msh", field_name: str | None):
    fields = result_mesh.field
    if field_name is not None:
        return field_name, fields[field_name]
    for name in _COMMON_FIELDS:
        if name in fields:
            return name, fields[name]
    raise KeyError(
        f"no known field found on result mesh; available: {list(fields.keys())}. "
        f"Pass field_name explicitly."
    )


def extract_axis_fields(
    result_mesh: "mesh_io.Msh",
    placement: "SEEGPlacement",
    field_name: str | None = None,
    grid_step_mm: float = 0.05,
    radial_max_mm: float = 1.0,
    n_azimuth: int = 8,
    background_ring_mm: float = 2.0,
    catalog: "ElectrodeCatalog | None" = None,
) -> dict[str, AxisProfile]:
    """Sample ``result_mesh``'s field along/around each lead in ``placement``.

    Parameters
    ----------
    field_name : name of the field to sample; auto-detected among common names if None.
        Vector fields are reduced to their magnitude.
    grid_step_mm : spacing of the along-axis / radial sample points.
    radial_max_mm : maximum radial distance from the contact surface to sample.
    n_azimuth : radial rays are averaged over this many azimuthal directions.
    background_ring_mm : radial distance whose value defines the far-field reference.
    catalog : electrode catalog for geometry lookup (default = bundled catalog).
    """
    from simnibs.mesh_tools.mesh_io import ElementData
    from .catalog import ElectrodeCatalog

    if catalog is None:
        catalog = ElectrodeCatalog.default()

    name, fobj = _pick_field(result_mesh, field_name)
    vals = np.asarray(fobj.value)
    is_vector = vals.ndim == 2 and vals.shape[1] > 1
    is_element = isinstance(fobj, ElementData)

    # exclude the electrode body (metal contact + insulating shaft) from the
    # interpolation -- read the field in tissue / glial sheath, never in the
    # conductor or insulator. The sheath IS tissue we want to measure, so it stays.
    tet_mask = result_mesh.elm.get_tetrahedra()
    body = tet_mask & np.isin(result_mesh.elm.tag1, [SEEG_CONTACT, SEEG_SHAFT])
    keep = tet_mask & ~body
    th_indices = result_mesh.elm.elm_number[keep]  # 1-based element numbers

    def sample(points: np.ndarray) -> np.ndarray:
        if is_element:
            interp = fobj.interpolate_scattered(
                points, out_fill=np.nan, method="assign", th_indices=th_indices
            )
        else:  # NodeData has no `method` kwarg
            interp = fobj.interpolate_scattered(
                points, out_fill=np.nan, th_indices=th_indices
            )
        interp = np.asarray(interp)
        if is_vector and interp.ndim == 2 and interp.shape[1] > 1:
            return np.linalg.norm(interp, axis=1)
        return interp.reshape(-1)

    out: dict[str, AxisProfile] = {}
    radial_mm = np.arange(grid_step_mm, radial_max_mm + 1e-9, grid_step_mm)
    azimuths = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)

    for lead in placement.leads:
        spec = catalog[lead.part_number]
        u = lead.axis_unit()
        e1, e2 = _perp_frame(u)
        centers = placement.contact_centers[lead.name]
        n_contacts = centers.shape[0]
        r_c = spec.contact_radius_mm
        r_off = r_c + max(2 * grid_step_mm, 0.1)  # axial line just outside the body

        # --- assemble ALL sample points for this lead, then one interpolation ---
        blocks = []

        # (1) axial line: parallel to axis, offset r_off into tissue (span = actual array)
        span = lead.first_contact_depth_mm + (n_contacts - 1) * spec.spacing_mm
        s = np.arange(0.0, span + spec.spacing_mm, grid_step_mm)
        axis_pts = lead.B[None, :] - u[None, :] * s[:, None] + e1[None, :] * r_off
        blocks.append(axis_pts)
        n_axial = axis_pts.shape[0]

        # (2) radial rays from each contact, all azimuths
        radii = r_c + radial_mm
        n_rad = radial_mm.size
        for c in centers:
            for phi in azimuths:
                dirv = np.cos(phi) * e1 + np.sin(phi) * e2
                blocks.append(c[None, :] + dirv[None, :] * radii[:, None])

        # (3) background ring at background_ring_mm from the mid contact
        cmid = centers[n_contacts // 2]
        ring = np.array(
            [cmid + (np.cos(phi) * e1 + np.sin(phi) * e2) * (r_c + background_ring_mm)
             for phi in azimuths]
        )
        blocks.append(ring)

        allpts = np.concatenate(blocks, axis=0)
        allval = sample(allpts)  # single interpolate_scattered call

        # --- slice results back ---
        off = 0
        axial_values = allval[off:off + n_axial]; off += n_axial
        radial_values = np.full((n_contacts, n_rad), np.nan)
        for i in range(n_contacts):
            acc = np.empty((n_azimuth, n_rad))
            for j in range(n_azimuth):
                acc[j] = allval[off:off + n_rad]; off += n_rad
            radial_values[i] = np.nanmean(acc, axis=0)
        background = float(np.nanmedian(allval[off:off + n_azimuth])); off += n_azimuth

        out[lead.name] = AxisProfile(
            lead_name=lead.name,
            contact_centers=centers,
            axial_mm=s,
            axial_values=axial_values,
            radial_mm=radial_mm,
            radial_values=radial_values,
            background=background,
        )
    return out
