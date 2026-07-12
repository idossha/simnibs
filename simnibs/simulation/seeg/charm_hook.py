"""Native charm hook: optionally embed sEEG electrode compartments into the head mesh.

This is the *single* integration point between the sEEG module and the standard charm
pipeline (``segmentation/charm_main.run``). It is a no-op unless the user supplies a
``--seeg <spec.json>`` file: charm then paints the sEEG compartments -- metallic contact
(tag 13), insulating shaft (tag 14) and the host-keyed peri-electrode sheath (glial sheath 15
in brain GM/WM, fibrous sheath 16 in the bone/scalp/soft-tissue tract; bare in CSF) -- into
charm's own upsampled tissue-label image *before* ``create_mesh`` runs, and augments charm's
meshing settings with a fine element size for those tags. Everything else in charm is
unchanged, so the head stays native SimNIBS quality and the electrodes are just an extra,
optional layer.

The four tags live in the free tissue band of ``ElementTags`` (< 100), so the standard
solver / ``cond_utils.standard_cond`` treat them as ordinary tissues -- no solver changes.

Off by default (``seeg=None``); on only when electrode locations are supplied.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from ._params import (
    SEEG_TAGS,
    SEEG_HIERARCHY,
    SEEGMaterials,
    ConformingResolution,
    CONF_FINE,
    augment_electrode_sizes,
    sheath_paint_warnings,
)
from .catalog import ElectrodeCatalog
from .geometry import SEEGLead
from .conductivity import ensure_seeg_registered, build_seeg_cond_list
from . import _raster

logger = logging.getLogger("simnibs.seeg")

# Electrode-first hierarchy so the electrode/tissue interface keeps a clean 1013-1016 shell.
# Single source of truth in _params (== meshing.create_mesh's native default with the sEEG
# tags 13-16 prepended); aliased here for back-compat with importers/tests.
_HIERARCHY = SEEG_HIERARCHY


@dataclass
class SeegSpec:
    """Parsed ``--seeg`` specification: leads + sheath/mesh GEOMETRY only.

    charm does segmentation + meshing; it does not take conductivities. The tags are meshed
    and registered with the *default* sEEG conductivities; the sheath conductivity (and any
    sweep) is a simulation-time input, set on the built mesh at solve time (see
    ``load_seeg_cond_list``). The sheath is meshed at its true ``sheath_thickness_um`` -- resolve
    it with a fine enough ``label_voxel_mm`` (there is no conductivity scaling to fake a
    sub-voxel sheath).
    """

    leads: list[SEEGLead]
    sheath_thickness_um: float = 150.0
    resolution: ConformingResolution = CONF_FINE
    catalog: ElectrodeCatalog | None = None

    def resolved_catalog(self) -> ElectrodeCatalog:
        return self.catalog if self.catalog is not None else ElectrodeCatalog.default()


def load_seeg_spec(spec) -> SeegSpec:
    """Parse a ``--seeg`` JSON file (or an already-loaded dict) into a :class:`SeegSpec`.

    Expected JSON::

        {
          "leads": [
            {"name": "R.AMY", "part_number": "BF10R-SP21X-0C3",
             "entry_mm": [x, y, z], "target_mm": [x, y, z],
             "n_contacts": 5, "first_contact_depth_mm": 2.0}
          ],
          "sheath_thickness_um": 150,        # optional (default 150); sheath GEOMETRY, e.g. 100
          "label_voxel_mm": 0.25,            # optional (electrode geometric fidelity)
          "electrode_edge_mm": 0.12,         # optional (fine tet size at the electrode)
          "electrode_facet_distance_mm": 0.03,
          "catalog": null                    # optional path to a custom electrode catalog
        }

    Only ``leads`` is required; every other key falls back to the study defaults. There is
    deliberately NO ``materials`` / conductivity key: charm only segments + meshes and assigns
    the default sEEG conductivities; set sheath sigma (and sweeps) at simulation time.
    """
    if isinstance(spec, (str, os.PathLike)):
        with open(spec) as fh:
            d = json.load(fh)
    elif isinstance(spec, dict):
        d = dict(spec)
    else:
        raise TypeError(f"seeg spec must be a path or dict, got {type(spec).__name__}")

    leads_raw = d.get("leads")
    if not leads_raw:
        raise ValueError("seeg spec has no 'leads' (nothing to embed)")

    leads: list[SEEGLead] = []
    for L in leads_raw:
        try:
            leads.append(
                SEEGLead(
                    name=L["name"],
                    part_number=L["part_number"],
                    entry_mm=tuple(float(v) for v in L["entry_mm"]),
                    target_mm=tuple(float(v) for v in L["target_mm"]),
                    first_contact_depth_mm=float(L.get("first_contact_depth_mm", 2.0)),
                    n_contacts=L.get("n_contacts"),
                )
            )
        except KeyError as e:
            raise ValueError(f"seeg lead {L.get('name', '?')!r} missing field {e}") from None

    if "materials" in d:
        logger.warning(
            "seeg spec 'materials' is ignored: charm assigns default sEEG conductivities; "
            "set sheath sigma at simulation time (see load_seeg_cond_list)."
        )

    resolution = ConformingResolution(
        label_voxel_mm=float(d.get("label_voxel_mm", CONF_FINE.label_voxel_mm)),
        electrode_edge_mm=float(d.get("electrode_edge_mm", CONF_FINE.electrode_edge_mm)),
        electrode_facet_distance_mm=float(
            d.get("electrode_facet_distance_mm", CONF_FINE.electrode_facet_distance_mm)
        ),
        bulk_edge_mm=CONF_FINE.bulk_edge_mm,
        reach_mm=CONF_FINE.reach_mm,
    )

    catalog = ElectrodeCatalog(d["catalog"]) if d.get("catalog") else None
    return SeegSpec(
        leads=leads,
        sheath_thickness_um=float(d.get("sheath_thickness_um", 150.0)),
        resolution=resolution,
        catalog=catalog,
    )


@dataclass
class SeegHookResult:
    """What the hook feeds back to charm's ``create_mesh`` call, plus provenance."""

    label_buffer: np.ndarray
    label_affine: np.ndarray
    elem_sizes: dict
    facet_distances: dict
    hierarchy: tuple
    materials: SEEGMaterials              # DEFAULT sEEG conductivities (no scaling)
    leads: list[SEEGLead]
    sheath_thickness_mm: float            # sheath thickness requested (geometry)
    voxel_counts: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def sidecar(self) -> dict:
        """JSON-serialisable provenance written next to the mesh.

        charm assigns the **default** sEEG conductivities (no scaling). ``cond_list`` is the
        ready-to-use ``cond_list[tag-1]`` vector for those defaults (feed it via
        :func:`~simnibs.simulation.seeg.conductivity.load_seeg_cond_list`); pass your own
        ``materials`` to that helper to solve/sweep a different sheath sigma.
        """
        return {
            "leads": [
                {
                    "name": ld.name,
                    "part_number": ld.part_number,
                    "entry_mm": list(ld.entry_mm),
                    "target_mm": list(ld.target_mm),
                    "n_contacts": ld.n_contacts,
                }
                for ld in self.leads
            ],
            "sheath_thickness_mm": self.sheath_thickness_mm,
            "materials": self.materials.as_tag_map(),
            "cond_list": build_seeg_cond_list(materials=self.materials),
            "voxel_counts": {int(k): int(v) for k, v in self.voxel_counts.items()},
            "warnings": self.warnings,
        }


def apply_seeg_to_label(
    label_buffer: np.ndarray,
    label_affine: np.ndarray,
    spec,
    *,
    elem_sizes: dict,
    facet_distances: dict,
    hierarchy=None,
    displace_tags: tuple[int, ...] | None = None,
    sheath_tag_map: dict[int, int] | None = None,
) -> SeegHookResult:
    """Paint sEEG compartments into charm's cropped label and return updated mesh inputs.

    Parameters
    ----------
    label_buffer, label_affine
        charm's already-cropped tissue-label image and its affine (from
        ``tissue_labeling_upsampled`` after ``crop_vol``).
    spec
        a :class:`SeegSpec`, a path to a ``--seeg`` JSON, or a spec dict.
    elem_sizes, facet_distances
        charm's own meshing settings (``settings['mesh']``); returned augmented with a fine
        entry for the sEEG tags 13-16 (charm's tissue entries are left untouched, so the head
        is meshed exactly as native SimNIBS -- only the electrode is finely sized).
    sheath_tag_map
        host tissue -> sheath tag (segmented sheath); default ``_params.SHEATH_TAG_BY_HOST``
        (brain -> glial 15, bone/scalp/soft -> fibrous 16, CSF/blood -> bare).
    hierarchy
        charm's hierarchy setting (usually ``False``/``None`` -> native default). The three
        electrode tags are prepended so the electrode surfaces win twin facets; if charm
        passes a non-empty custom order it is preserved *after* the electrode tags rather
        than dropped.

    Returns
    -------
    SeegHookResult with the upsampled/painted label, the augmented settings and provenance.
    The four sEEG conductivities are registered via ``ensure_seeg_registered`` so the stock
    solver picks them up.
    """
    if not isinstance(spec, SeegSpec):
        spec = load_seeg_spec(spec)

    catalog = spec.resolved_catalog()
    materials = SEEGMaterials()          # charm assigns the DEFAULT conductivities (no spec sigma)
    resolution = spec.resolution
    sheath_mm = max(0.0, spec.sheath_thickness_um) / 1000.0
    warnings: list[str] = []

    # --- upsample charm's label to the electrode-fidelity voxel size ---
    # Only the electrode benefits from the finer label (its surface hugs the voxel staircase
    # to ~0.5*voxel); the tissue tet sizes stay native via charm's own elem_sizes.
    dv = resolution.label_voxel_mm
    label = np.round(label_buffer).astype(np.int16)
    label, affine = _raster.resample_iso(label, np.asarray(label_affine, float), dv)
    logger.info(
        "sEEG: upsampled charm label to %.2f mm -> %s (%.0f M voxels) for electrode fidelity",
        dv, label.shape, label.size / 1e6,
    )

    # --- fidelity + sheath-consistency guards (identical policy to build_conforming_head) ---
    min_body = min(catalog[ld.part_number].body_radius_mm * 2 for ld in spec.leads)
    if min_body / dv < 4.0:
        msg = (f"contact/body spans only ~{min_body / dv:.1f} voxels at {dv:.2f} mm; "
               f"use a smaller label_voxel_mm for a rounder rod.")
        logger.warning(msg); warnings.append(msg)

    # --- sheath resolution guard (geometry only; NO conductivity scaling) ---
    # The sheath is meshed at its true thickness. If it is thinner than the label voxel it
    # cannot be resolved -- the sheath conductivity is NEVER scaled to fake it; use a finer
    # label_voxel_mm (or a surface-conforming build) to mesh a true sub-voxel sheath.
    if 0.0 < sheath_mm < dv:
        msg = (f"sheath {sheath_mm * 1000:.0f} um < label voxel {dv * 1000:.0f} um: it cannot "
               f"be meshed at true thickness -- set label_voxel_mm <= {sheath_mm * 1000:.0f} um "
               f"(the electrode is an analytic model, so a fine label is legitimate).")
        logger.warning(msg); warnings.append(msg)

    # --- paint the electrodes into the real tissue label (no background block) ---
    label, vox_counts = _raster.paint_leads_into_label(
        label, affine, spec.leads, catalog, sheath_mm,
        displace_tags=displace_tags, sheath_tag_map=sheath_tag_map,
    )
    logger.info("sEEG: painted voxels %s", vox_counts)
    for msg in sheath_paint_warnings(vox_counts):
        logger.warning(msg); warnings.append(msg)

    label = label.astype(np.uint16 if label.max() >= 256 else np.uint8)

    # --- augment charm's meshing settings with a fine electrode; tissues untouched ---
    el = dict(elem_sizes)
    fd = dict(facet_distances)
    augment_electrode_sizes(
        el, fd,
        float(resolution.electrode_edge_mm),
        float(resolution.electrode_facet_distance_mm),
    )

    # --- electrode-first hierarchy, preserving any custom order charm passed in ---
    # Prepend the sEEG tags so their surfaces win twin facets; keep the caller's tissue order
    # when one was supplied (charm's default is falsy -> the native SEEG_HIERARCHY).
    resolved_hierarchy = SEEG_HIERARCHY if not hierarchy else (*SEEG_TAGS, *hierarchy)

    # --- register the default conductivities for the stock solver ---
    ensure_seeg_registered(materials)

    return SeegHookResult(
        label_buffer=label,
        label_affine=affine,
        elem_sizes=el,
        facet_distances=fd,
        hierarchy=resolved_hierarchy,
        materials=materials,
        leads=list(spec.leads),
        sheath_thickness_mm=sheath_mm,
        voxel_counts={int(k): int(v) for k, v in vox_counts.items()},
        warnings=warnings,
    )
