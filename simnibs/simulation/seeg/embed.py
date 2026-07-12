"""``build_conforming_head`` -- rebuild the whole head with the sEEG electrodes as
*conforming* subdomains (smooth cylinders, not a tet staircase).

The electrode materials (contact=13 / shaft=14 / glial sheath=15 / fibrous sheath=16) are painted into the
subject's real tissue label image (replacing the tissue they occupy -- no background
block), the head is upsampled to the target voxel size, and the whole thing is remeshed
with ``meshing.create_mesh`` so the electrode surfaces are honoured. The result is a
run_simnibs-ready ``SEEGPlacement`` carrying the tagged mesh + conductivity list.

This is the whole-head remesh path (tens of minutes to a couple of hours). It mirrors what the
native ``charm --seeg`` step does (paint the same compartments, then create_mesh with charm's
own tissue sizing) as a standalone entry point that does not need the charm CLI; the native
step itself uses ``charm_hook.apply_seeg_to_label`` inside charm's mesh block, not this
function. See ``ConformingResolution`` presets for the fidelity/cost tradeoff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ._params import (
    SEEG_TAGS,
    SEEG_CONTACT,
    SEEG_SHAFT,
    GLIAL_SHEATH,
    FIBROUS_SHEATH,
    SEEG_HIERARCHY,
    SEEGMaterials,
    ConformingResolution,
    CONF_FINE,
    sheath_paint_warnings,
)
from .catalog import ElectrodeCatalog
from .geometry import SEEGLead, perp_distance_and_axial
from .conductivity import ensure_seeg_registered, build_seeg_cond_list
from . import _raster

if TYPE_CHECKING:  # pragma: no cover
    from simnibs.mesh_tools import mesh_io

logger = logging.getLogger("simnibs.seeg")


@dataclass
class SEEGPlacement:
    """Result of :func:`build_conforming_head` -- the tagged head mesh + a ready-to-use
    conductivity list, plus per-lead contact geometry for downstream sampling."""

    mesh: "mesh_io.Msh"
    leads: list[SEEGLead]
    materials: SEEGMaterials
    sheath_thickness_mm: float
    method: str = "conforming"
    contact_centers: dict[str, np.ndarray] = field(default_factory=dict)
    contact_axes: dict[str, np.ndarray] = field(default_factory=dict)
    tag_counts: dict[int, int] = field(default_factory=dict)
    cond_list: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_contact_tets(self) -> int:
        return int(self.tag_counts.get(SEEG_CONTACT, 0))

    @property
    def n_shaft_tets(self) -> int:
        return int(self.tag_counts.get(SEEG_SHAFT, 0))

    @property
    def n_sheath_tets(self) -> int:
        return int(self.tag_counts.get(GLIAL_SHEATH, 0))

    @property
    def n_fibrous_tets(self) -> int:
        return int(self.tag_counts.get(FIBROUS_SHEATH, 0))

    def summary(self) -> str:
        lines = [
            f"SEEGPlacement [{self.method}]: {len(self.leads)} lead(s), "
            f"sheath {self.sheath_thickness_mm * 1000:.0f} um",
            f"  contact tets: {self.n_contact_tets}",
            f"  shaft   tets: {self.n_shaft_tets}",
            f"  glial sheath  tets: {self.n_sheath_tets}",
            f"  fibrous sheath tets: {self.n_fibrous_tets}",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

def build_conforming_head(
    reference,
    leads: list[SEEGLead],
    *,
    sheath_thickness_um: float = 150.0,
    materials: SEEGMaterials | None = None,
    catalog: ElectrodeCatalog | None = None,
    resolution: ConformingResolution | None = None,
    displace_tags: tuple[int, ...] | None = None,
    sheath_tag_map: dict[int, int] | None = None,
    mesher: str = "create_mesh",
    num_threads: int = 8,
    crop_to_head: bool = True,
) -> SEEGPlacement:
    """Rebuild the whole head mesh with conforming sEEG electrodes.

    Parameters
    ----------
    reference : str | os.PathLike | (label_array, affine)
        m2m folder (uses ``final_tissues.nii.gz``), a tissue-label ``.nii.gz``, or a
        ``(label_image, affine)`` tuple.
    leads : list[SEEGLead]
        Electrodes; ``part_number`` selects geometry from ``catalog``.
    sheath_thickness_um : sheath thickness (um), same for both sheath types.
    materials : conductivities for tags 13/14/15/16 (default = stable defaults).
    catalog : electrode geometry lookup (default = bundled catalog).
    resolution : ConformingResolution (default = CONF_FINE; label_voxel_mm sets fidelity).
    displace_tags : tissue tags the body may replace (None -> 1..12).
    sheath_tag_map : host tissue -> sheath tag (segmented sheath); None -> the default
        ``_params.SHEATH_TAG_BY_HOST`` (brain -> glial 15, bone/scalp/soft -> fibrous 16,
        CSF/blood -> bare shaft).
    mesher : "create_mesh" (default, solver-ready) | "image2mesh" (advanced/raw CGAL).
    num_threads : CGAL threads.
    crop_to_head : crop to the nonzero-tissue bbox before upsampling (big RAM/time saving).

    Returns
    -------
    SEEGPlacement with ``method="conforming"``.
    """
    from simnibs.mesh_tools import meshing

    if materials is None:
        materials = SEEGMaterials()
    if catalog is None:
        catalog = ElectrodeCatalog.default()
    if resolution is None:
        resolution = CONF_FINE
    if not leads:
        raise ValueError("no leads given")

    sheath_mm = max(0.0, sheath_thickness_um) / 1000.0
    warnings: list[str] = []

    # --- resolve + crop + upsample the head label image ---
    label, affine = _raster.resolve_reference(reference)
    if crop_to_head:
        label, affine = _raster.crop_nonzero(label, affine, pad_mm=6.0)
    label, affine = _raster.resample_iso(label, affine, resolution.label_voxel_mm)
    logger.info(
        "sEEG conforming: label grid %s @ %.2f mm (%.0f M voxels)",
        label.shape, resolution.label_voxel_mm, label.size / 1e6,
    )

    dv = resolution.label_voxel_mm
    # --- fidelity warnings (honest limits of a uniform grid) ---
    min_body = min(catalog[ld.part_number].body_radius_mm * 2 for ld in leads)
    if min_body / dv < 4.0:
        msg = (f"contact/body spans only ~{min_body/dv:.1f} voxels at {dv:.2f} mm; "
               f"use a finer label_voxel_mm (CONF_ULTRA) for a rounder rod.")
        logger.warning(msg); warnings.append(msg)

    # Sheath consistency: a sub-voxel sheath paints intermittently (inconsistent). Grow the
    # meshed sheath to >=1 voxel so it is a CONSISTENT annulus, keeping the sheet resistance
    # R_s = t/sigma by scaling sigma with the thickened t (thin-shell equivalence).
    paint_sheath_mm = sheath_mm
    if sheath_mm > 0 and sheath_mm < dv:
        paint_sheath_mm = float(dv)
        scale = paint_sheath_mm / sheath_mm
        materials = materials.with_scaled_sheath(scale)   # keep both shells' t/sigma constant
        msg = (f"sheath {sheath_mm*1000:.0f} um < voxel {dv*1000:.0f} um: meshed at "
               f"1 voxel with sigma scaled x{scale:.2f} (preserves t/sigma). For a "
               f"true-thickness sheath use preview_electrode / the surface route.")
        logger.warning(msg); warnings.append(msg)

    # --- paint the electrodes into the real tissue label ---
    label, vox_counts = _raster.paint_leads_into_label(
        label, affine, leads, catalog, paint_sheath_mm,
        displace_tags=displace_tags, sheath_tag_map=sheath_tag_map,
    )
    logger.info("sEEG conforming: painted voxels %s", vox_counts)
    for msg in sheath_paint_warnings(vox_counts):
        logger.warning(msg); warnings.append(msg)

    # CGAL meshers want an integer label image
    if label.max() < 256:
        label = label.astype(np.uint8)
    else:
        label = label.astype(np.uint16)

    # --- remesh the whole head, conforming to the electrode surfaces ---
    if mesher == "create_mesh":
        # Native charm pipeline: per-tissue elem_sizes reproduce the standard SimNIBS head;
        # smoothing/skin-care match charm; only the electrode is added (fine). This keeps the
        # head native and just adds the sEEG layer (heavier: create_mesh preprocessing is
        # voxel-bound, so a fine label runs longer than the image2mesh fast path).
        cm = resolution.to_create_mesh_kwargs()
        m = meshing.create_mesh(
            label, affine,
            elem_sizes=cm["elem_sizes"], facet_distances=cm["facet_distances"],
            hierarchy=SEEG_HIERARCHY, skin_tag=1005, apply_cream=True,
            smooth_steps=10, skin_care=20, skin_facet_size=2.0,
            optimize=False, num_threads=num_threads,
        )
    elif mesher == "image2mesh":
        m = _image2mesh_head(meshing, label, affine, leads, catalog, sheath_mm,
                             resolution, num_threads)
    else:
        raise ValueError(f"mesher must be 'create_mesh'|'image2mesh', got {mesher!r}")

    # --- assemble the method-agnostic result ---
    ensure_seeg_registered(materials)
    tet = m.elm.get_tetrahedra()
    tag1 = m.elm.tag1
    contact_centers = {ld.name: ld.contact_centers(catalog[ld.part_number]) for ld in leads}
    contact_axes = {
        ld.name: np.tile(ld.axis_unit(), (contact_centers[ld.name].shape[0], 1))
        for ld in leads
    }
    tag_counts = {int(t): int(((tag1 == t) & tet).sum()) for t in SEEG_TAGS}
    for msg in sheath_paint_warnings(tag_counts):
        logger.warning(msg)
        if msg not in warnings:
            warnings.append(msg)

    return SEEGPlacement(
        mesh=m,
        leads=list(leads),
        materials=materials,
        sheath_thickness_mm=sheath_mm,
        method="conforming",
        contact_centers=contact_centers,
        contact_axes=contact_axes,
        tag_counts=tag_counts,
        cond_list=build_seeg_cond_list(materials=materials),
        warnings=warnings,
    )


def _radial_sizing_field(label, affine, leads, catalog, resolution):
    """Per-voxel target size + facet-distance fields: fine at the electrode, coarse in bulk.

    Fine (``electrode_edge_mm`` / ``electrode_facet_distance_mm``) within the electrode body,
    ramping linearly to (``bulk_edge_mm`` / bulk facet distance) over ``reach_mm`` from the
    axis; coarse everywhere else. Keeps the whole-head tet count small while resolving the
    electrode. Returned as float32 F-contiguous arrays for ``cgal.mesh_image_sizing_field``.
    """
    e = resolution.electrode_edge_mm
    b = resolution.bulk_edge_mm
    efd = resolution.electrode_facet_distance_mm
    bfd = 1.0                                    # coarse boundary tolerance in the bulk
    reach = resolution.reach_mm
    size = np.full(label.shape, b, dtype=np.float32)
    fdist = np.full(label.shape, bfd, dtype=np.float32)
    inv = np.linalg.inv(affine)
    for lead in leads:
        spec = catalog[lead.part_number]
        A, B = lead.A, lead.B
        r_body = spec.body_radius_mm
        r_out = r_body + reach
        lo, hi = _raster._seg_voxel_bbox(A, B, r_out + 1.0, inv, label.shape)
        if np.any(hi <= lo):
            continue
        P = _raster._voxel_centres_world(affine, lo, hi - lo)
        d, _ = perp_distance_and_axial(P, A, B)
        frac = np.clip((d - r_body) / max(reach, 1e-6), 0.0, 1.0)
        sz = (e + frac * (b - e)).reshape(hi - lo).astype(np.float32)
        fd = (efd + frac * (bfd - efd)).reshape(hi - lo).astype(np.float32)
        sub = size[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        fsub = fdist[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.minimum(sub, sz, out=sub)             # keep the finest where leads overlap
        np.minimum(fsub, fd, out=fsub)
    return np.asfortranarray(size), np.asfortranarray(fdist)


def _image2mesh_head(meshing, label, affine, leads, catalog, sheath_mm, resolution, num_threads):
    """Raw CGAL image2mesh with a radial sizing field (fine only at the electrode) + remap.

    A uniform-fine facet size over a whole-head crop OOMs (M1); the sizing field puts small
    tets/facets ONLY near the electrode, coarse elsewhere, so the label can stay fine
    (sub-mm electrode geometry) while the tet count stays small.
    """
    size, fdist = _radial_sizing_field(label, affine, leads, catalog, resolution)
    logger.info("sEEG conforming: image2mesh, sizing %.2f-%.2f mm, fine facets at electrode",
                float(size.min()), float(size.max()))
    m = meshing.image2mesh(
        label, affine, facet_angle=30,
        facet_size=size, facet_distance=fdist,
        cell_radius_edge_ratio=3, cell_size=size,
        num_threads=num_threads,
    )
    # Undo CGAL's ascending 1..N renumbering with the library routine (keeps largest-count
    # labels; a hand-rolled sorted-unique map would mislabel every tag above a dropped region).
    meshing._fix_labels(m, label)
    ensure_head_surfaces(m)
    return m


def ensure_head_surfaces(m: "mesh_io.Msh", skin_vol_tag: int = 5) -> None:
    """Reconstruct volume-boundary surfaces (1000+tag), guaranteeing the 1005 skin."""
    vol_tags = [int(t) for t in np.unique(m.elm.tag1[m.elm.get_tetrahedra()])]
    m.reconstruct_surfaces(tags=vol_tags)
