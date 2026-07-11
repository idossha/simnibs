"""Build a *conforming* (smooth) sEEG electrode mesh via CGAL image meshing.

A whole-head conforming build is bounded by the head label voxel size. For a genuinely
smooth *reference* electrode -- a clean cylindrical rod and a clean sheath annulus --
rasterise the
electrode analytically (in its own axis-aligned frame, so the image is tiny) and let
CGAL mesh it with a surface that *conforms* to the cylinders (``facet_distance``
controls smoothness). This mirrors the CAD-conforming meshes used for implant field
studies (Karimi et al. 2025).

The result is a **standalone** electrode mesh (contact/shaft/sheath, world coords) for
visualisation / QC / a smooth single-electrode figure. To embed conforming electrodes in
a solvable head, use :func:`~simnibs.simulation.seeg.embed.build_conforming_head`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ._params import SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH
from .geometry import SEEGLead

if TYPE_CHECKING:  # pragma: no cover
    from simnibs.mesh_tools import mesh_io
    from .catalog import ElectrodeCatalog, ElectrodeSpec

logger = logging.getLogger("simnibs.seeg")

_BACKGROUND_TAG = 2  # grey matter, so the electrode sits in a brain-like block


def preview_electrode(
    lead: SEEGLead,
    catalog: "ElectrodeCatalog | None" = None,
    spec: "ElectrodeSpec | None" = None,
    sheath_thickness_um: float = 200.0,
    voxel_um: float = 40.0,
    facet_distance_um: float = 20.0,
    cell_size_mm: float = 0.12,
    perp_margin_mm: float = 0.6,
    num_threads: int = 8,
) -> "mesh_io.Msh":
    """Return a smooth, conforming *standalone* mesh of one sEEG lead (figures / QC).

    This wraps the electrode in a grey-matter background block (a "preview box"); it is
    NOT embedded in a head. To put conforming electrodes into a solvable head, use
    :func:`~simnibs.simulation.seeg.embed.build_conforming_head`.

    Parameters
    ----------
    lead : SEEGLead
        Trajectory (entry -> tip) of the electrode.
    catalog / spec : geometry source (``spec`` wins; else ``catalog[lead.part_number]``;
        else the bundled catalog).
    sheath_thickness_um : glial-sheath thickness (µm).
    voxel_um : rasterisation voxel size (µm) — the geometric fidelity of the label image.
    facet_distance_um : CGAL facet distance (µm) — how closely surfaces follow the true
        cylinders; smaller = smoother (default 20 µm).
    cell_size_mm : CGAL target tetrahedron size (mm).
    perp_margin_mm : padding around the sheath radius in the perpendicular plane.

    Returns
    -------
    mesh_io.Msh with tetrahedra tagged SEEG_CONTACT(13) / SEEG_SHAFT(14) /
    GLIAL_SHEATH(15) plus background GM(2), in world/RAS mm.
    """
    from simnibs.mesh_tools import meshing, mesh_io

    if spec is None:
        if catalog is None:
            from .catalog import ElectrodeCatalog

            catalog = ElectrodeCatalog.default()
        spec = catalog[lead.part_number]

    A = lead.A
    u = lead.axis_unit()
    L = lead.length_mm
    # perpendicular frame
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    centers = lead.contact_centers(spec)
    s_c = (centers - A) @ u                      # contact axial coords (from entry)
    r_c = spec.contact_radius_mm
    r_s = spec.shaft_radius_mm
    r_sheath = max(r_c, r_s) + max(0.0, sheath_thickness_um) / 1000.0
    hlen = 0.5 * spec.contact_len_mm

    dv = voxel_um / 1000.0
    half = r_sheath + perp_margin_mm
    lo_perp = -half
    lo_s, hi_s = -2.0, L + 2.0
    n_perp = int(np.ceil(2 * half / dv))
    n_s = int(np.ceil((hi_s - lo_s) / dv))

    x = lo_perp + dv * np.arange(n_perp)
    s = lo_s + dv * np.arange(n_s)
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    dgrid = np.sqrt(X1 ** 2 + X2 ** 2)                       # (n_perp, n_perp)
    near = np.min(np.abs(s[:, None] - s_c[None, :]), axis=1) <= hlen  # (n_s,)
    in_shaft = (s >= s_c.min() - hlen) & (s <= L)

    lab = np.full((n_perp, n_perp, n_s), _BACKGROUND_TAG, dtype=np.uint8)
    D = dgrid[:, :, None]
    metal = (D <= r_c) & near[None, None, :]
    shaft = (D <= r_s) & ~metal & in_shaft[None, None, :]
    inner = np.where(near, r_c, r_s)[None, None, :]
    sheath = (D > inner) & (D <= r_sheath) & (near | in_shaft)[None, None, :]
    lab[sheath] = GLIAL_SHEATH
    lab[shaft] = SEEG_SHAFT
    lab[metal] = SEEG_CONTACT

    # affine: image axes (i, j, k) -> world (e1, e2, u)
    affine = np.eye(4)
    affine[:3, 0] = dv * e1
    affine[:3, 1] = dv * e2
    affine[:3, 2] = dv * u
    affine[:3, 3] = A + lo_perp * e1 + lo_perp * e2 + lo_s * u

    logger.info(
        "sEEG CGAL electrode: label %s (%.1f M vox), facet_distance %.0f um",
        lab.shape, lab.size / 1e6, facet_distance_um,
    )
    m = meshing.image2mesh(
        lab, affine, facet_angle=30, facet_size=cell_size_mm,
        facet_distance=facet_distance_um / 1000.0,
        cell_radius_edge_ratio=3, cell_size=cell_size_mm, num_threads=num_threads,
    )

    # CGAL renumbers labels to 1..N (ascending). Re-identify by geometry.
    _relabel_by_geometry(m, A, u, e1, e2, s_c, r_c, r_s, hlen)
    return m


def _relabel_by_geometry(m, A, u, e1, e2, s_c, r_c, r_s, hlen) -> None:
    """Map CGAL's renumbered tags back to SEEG_CONTACT/SHAFT/SHEATH by geometry."""
    import numpy as np

    tet = m.elm.get_tetrahedra()
    bar = m.elements_baricenters().value
    rel = bar - A
    axial = rel @ u
    d = np.linalg.norm(rel - np.outer(axial, u), axis=1)
    near = np.min(np.abs(axial[:, None] - s_c[None, :]), axis=1) <= hlen

    tags = np.unique(m.elm.tag1[tet])
    new = m.elm.tag1.copy()
    r_body = max(r_c, r_s)
    for tg in tags:
        sel = tet & (m.elm.tag1 == tg)
        if sel.sum() == 0:
            continue
        md = np.median(d[sel])
        if md > r_body:            # annulus
            kind = GLIAL_SHEATH
        elif near[sel].mean() > 0.5:
            kind = SEEG_CONTACT
        else:
            kind = SEEG_SHAFT
        # the big background block (median d large AND covers most volume) stays GM
        if md > 1.5 * r_body and sel.sum() > 0.4 * tet.sum():
            kind = _BACKGROUND_TAG
        new[sel] = kind
    m.elm.tag1 = new
    m.elm.tag2 = new.copy()
