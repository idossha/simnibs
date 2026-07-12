"""Tests for the conforming embed method.

Rasteriser helpers are pure-numpy (fast). The full build runs CGAL image2mesh on a tiny
synthetic head (~5 s) and is skipped if the meshing backend is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from simnibs.simulation.seeg import (
    SEEGLead, ConformingResolution, CONF_FINE, CONF_STANDARD, CONF_ULTRA,
    SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH, FIBROUS_SHEATH,
)
from simnibs.simulation.seeg import _raster
from simnibs.simulation.seeg.catalog import ElectrodeCatalog


# --------------------------------------------------------------------------- #
# ConformingResolution
# --------------------------------------------------------------------------- #
def test_conforming_resolution_to_kwargs():
    r = ConformingResolution(0.25, 0.12, 0.03, 2.5, 3.0)
    kw = r.to_create_mesh_kwargs()
    for t in ("13", "14", "15"):
        assert kw["elem_sizes"][t]["range"] == [0.12, 0.12]
        assert kw["facet_distances"][t]["range"] == [0.03, 0.03]


def test_conforming_presets_fidelity_ladder():
    assert CONF_STANDARD.label_voxel_mm > CONF_FINE.label_voxel_mm > CONF_ULTRA.label_voxel_mm


# --------------------------------------------------------------------------- #
# rasteriser (pure numpy)
# --------------------------------------------------------------------------- #
def _block(n=220, vox=0.15):
    # voxel fine enough to resolve a 150 um sheath at true thickness (no widening): 0.15/0.15 = 1
    lab = np.zeros((n, n, n), np.uint8)
    lab[8:-8, 8:-8, 8:-8] = 2  # GM block
    aff = np.eye(4); aff[0, 0] = aff[1, 1] = aff[2, 2] = vox; aff[:3, 3] = -vox * n / 2
    return lab, aff


def test_paint_constant_radius_rod():
    lab, aff = _block()
    lead = SEEGLead("T", "BF10R-SP21X-0C3", entry_mm=(12, 0, 0), target_mm=(-12, 0, 0), n_contacts=5)
    out, counts = _raster.paint_leads_into_label(lab, aff, [lead], ElectrodeCatalog.default(), 0.15)
    vals = set(np.unique(out).tolist())
    assert {13, 14, 15}.issubset(vals)
    assert vals.issubset({0, 2, 13, 14, 15})              # no stray block tag
    assert all(counts[t] > 0 for t in (SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH))
    # constant outer radius: metal+shaft (the body) share one radius -> the rod cross-section
    # radius does not depend on axial position (contacts differ only by tag)
    spec = ElectrodeCatalog.default()["BF10R-SP21X-0C3"]
    bar_body = np.argwhere(np.isin(out, [13, 14]))
    P = (aff[:3, :3] @ bar_body.T + aff[:3, 3, None]).T
    # radial distance to the (infinite) electrode axis line -- the constant-radius check must be
    # purely radial, unaffected by the entry-extended shaft running past the original entry.
    u = lead.axis_unit()
    rel = P - lead.A
    axial = rel @ u
    d = np.linalg.norm(rel - np.outer(axial, u), axis=1)
    assert d.max() <= spec.body_radius_mm + aff[0, 0]     # within one voxel of the body radius


def test_paint_full_encapsulation_no_bare_gm_wm():
    """The electrode is fully wrapped: no metal/shaft voxel touches bare GM/WM."""
    from scipy.ndimage import binary_dilation
    lab, aff = _block()
    lab[:] = 2                                             # solid GM
    lead = SEEGLead("T", "BF10R-SP21X-0C3", entry_mm=(12, 0, 0), target_mm=(-12, 0, 0), n_contacts=5)
    out, counts = _raster.paint_leads_into_label(lab, aff, [lead], ElectrodeCatalog.default(), 0.15)
    body = np.isin(out, [13, 14])
    st = np.zeros((3, 3, 3), bool)
    for d in [(1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        st[d] = True
    # every tissue voxel face-adjacent to the body must be sheath (never bare GM=2)
    nb = binary_dilation(body, structure=st) & ~body
    assert int((nb & (out == 2)).sum()) == 0              # no bare GM contact
    assert int((nb & np.isin(out, list(range(1, 13))) & (out != 15)).sum()) == 0


def test_paint_segmented_sheath_tags_by_host():
    """The sheath is host-keyed: glial (15) in brain, fibrous (16) in bone, none in CSF."""
    lead = SEEGLead("T", "BF10R-SP21X-0C3", entry_mm=(12, 0, 0), target_mm=(-12, 0, 0), n_contacts=5)
    cat = ElectrodeCatalog.default()

    # brain (GM) -> glial sheath only
    lab, aff = _block(); lab[:] = 2
    _, c_gm = _raster.paint_leads_into_label(lab, aff, [lead], cat, 0.15)
    assert c_gm[GLIAL_SHEATH] > 0 and c_gm[FIBROUS_SHEATH] == 0

    # bone -> fibrous sheath only (the through-skull shunt), never glial
    lab, aff = _block(); lab[:] = 4
    _, c_bone = _raster.paint_leads_into_label(lab, aff, [lead], cat, 0.15)
    assert c_bone[FIBROUS_SHEATH] > 0 and c_bone[GLIAL_SHEATH] == 0
    assert c_bone[SEEG_CONTACT] > 0

    # CSF -> no sheath at all (bare shaft), but the electrode body is still painted
    lab, aff = _block(); lab[:] = 3
    _, c_csf = _raster.paint_leads_into_label(lab, aff, [lead], cat, 0.15)
    assert c_csf[GLIAL_SHEATH] == 0 and c_csf[FIBROUS_SHEATH] == 0
    assert c_csf[SEEG_CONTACT] > 0 and c_csf[SEEG_SHAFT] > 0

    # explicit override: a custom map can force sheath even in CSF
    _, c_map = _raster.paint_leads_into_label(
        lab, aff, [lead], cat, 0.15, sheath_tag_map={3: GLIAL_SHEATH})
    assert c_map[GLIAL_SHEATH] > 0


def test_resample_iso_alignment():
    lab, aff = _block(n=40, vox=1.0)
    out, aff2 = _raster.resample_iso(lab, aff, 0.5)
    assert set(np.unique(out).tolist()) == set(np.unique(lab).tolist())
    assert np.allclose(np.linalg.norm(aff2[:3, :3], axis=0), 0.5, atol=0.02)
    mid = np.array([20, 20, 20, 1.0]); w = aff @ mid
    v2 = np.linalg.inv(aff2) @ w
    assert np.linalg.norm((aff2 @ np.round(v2))[:3] - w[:3]) < 0.5


# --------------------------------------------------------------------------- #
# full build (needs meshing/CGAL)
# --------------------------------------------------------------------------- #
def test_build_conforming_head_synthetic():
    pytest.importorskip("simnibs.mesh_tools.meshing")
    from simnibs.simulation.seeg import build_conforming_head

    lab, aff = _block(n=180, vox=0.2)
    lead = SEEGLead("T", "BF10R-SP21X-0C3", entry_mm=(16, 0, 0), target_mm=(-16, 0, 0), n_contacts=5)
    pl = build_conforming_head(
        (lab, aff), [lead], sheath_thickness_um=150,
        resolution=ConformingResolution(0.2, 0.12, 0.03, 2.0, 2.5), num_threads=4,
    )
    assert pl.method == "conforming"
    tet = pl.mesh.elm.get_tetrahedra()
    vt = set(int(t) for t in np.unique(pl.mesh.elm.tag1[tet]))
    st = set(int(t) for t in np.unique(pl.mesh.elm.tag1[pl.mesh.elm.get_triangles()]))
    assert {13, 14, 15}.issubset(vt)
    assert {1013, 1014, 1015}.issubset(st)
    assert all(pl.tag_counts[t] > 0 for t in (SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH))
    assert pl.cond_list[SEEG_CONTACT - 1] == pytest.approx(1e6)
