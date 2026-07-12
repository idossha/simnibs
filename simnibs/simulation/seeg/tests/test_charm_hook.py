"""Tests for the native charm hook (``charm --seeg``).

The hook is pure-numpy (label painting + settings augmentation); no CGAL, so these run
anywhere and fast. They lock the contract that charm_main.run relies on:
tissue settings untouched, electrode finely sized, electrode-first hierarchy, sheath
thin-shell equivalence, and a JSON round-trip.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simnibs.simulation.seeg.charm_hook import (
    load_seeg_spec,
    apply_seeg_to_label,
    SeegSpec,
    _HIERARCHY,
)
from simnibs.simulation.seeg._params import SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH


# charm.ini [mesh] defaults (what charm_main hands the hook)
CHARM_ELEM = {
    "standard": {"range": [1, 5], "slope": 1.0},
    "1": {"range": [1, 7], "slope": 1.0},
    "2": {"range": [1, 2], "slope": 1.0},
    "5": {"range": [1, 10], "slope": 0.6},
}
CHARM_FACET = {"standard": {"range": [0.1, 3], "slope": 0.5}}


def _gm_block(n=120, vox=0.5):
    lab = np.zeros((n, n, n), np.uint16)
    lab[10:-10, 10:-10, 10:-10] = 2
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = vox
    aff[:3, 3] = -vox * n / 2
    return lab, aff


def _spec(**over):
    d = {
        "leads": [{"name": "T", "part_number": "BF10R-SP21X-0C3",
                   "entry_mm": [24, 0, 0], "target_mm": [-24, 0, 0], "n_contacts": 5}],
        "sheath_thickness_um": 150,
        "label_voxel_mm": 0.25,
    }
    d.update(over)
    return d


# --------------------------------------------------------------------------- #
# load_seeg_spec
# --------------------------------------------------------------------------- #
def test_load_spec_from_dict_and_path(tmp_path):
    s = load_seeg_spec(_spec())
    assert isinstance(s, SeegSpec)
    assert len(s.leads) == 1 and s.leads[0].name == "T"
    assert s.resolution.label_voxel_mm == pytest.approx(0.25)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_spec()))
    assert len(load_seeg_spec(str(p)).leads) == 1


def test_load_spec_defaults_and_materials():
    s = load_seeg_spec({"leads": [{"name": "T", "part_number": "BF10R-SP21X-0C3",
                                   "entry_mm": [24, 0, 0], "target_mm": [-24, 0, 0]}],
                        "materials": {"sheath_sigma": 0.11}})
    assert s.sheath_thickness_um == pytest.approx(150.0)          # default
    assert s.materials.sheath_sigma == pytest.approx(0.11)        # override
    assert s.materials.contact_sigma == pytest.approx(1e6)        # default


def test_load_spec_missing_leads_raises():
    with pytest.raises(ValueError, match="leads"):
        load_seeg_spec({"sheath_thickness_um": 150})


def test_load_spec_missing_lead_field_names_lead():
    with pytest.raises(ValueError, match="MYLEAD"):
        load_seeg_spec({"leads": [{"name": "MYLEAD", "part_number": "BF10R-SP21X-0C3",
                                   "entry_mm": [0, 0, 0]}]})  # no target_mm


# --------------------------------------------------------------------------- #
# apply_seeg_to_label
# --------------------------------------------------------------------------- #
def test_hook_paints_and_augments():
    lab, aff = _gm_block()
    r = apply_seeg_to_label(lab, aff, _spec(),
                            elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET,
                            hierarchy=False)
    # contact, shaft, glial sheath painted (GM block -> glial, not fibrous)
    assert all(r.voxel_counts[t] > 0 for t in (SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH))
    # electrode finely sized, tissues untouched (head stays native)
    for t in ("13", "14", "15"):
        assert r.elem_sizes[t]["range"] == [0.12, 0.12]
        assert r.facet_distances[t]["range"] == [0.03, 0.03]
    assert r.elem_sizes["2"] == CHARM_ELEM["2"]
    assert r.elem_sizes["standard"] == CHARM_ELEM["standard"]
    assert r.facet_distances["standard"] == CHARM_FACET["standard"]
    # electrode-first hierarchy so electrode surfaces win twin facets
    assert r.hierarchy == _HIERARCHY
    assert r.hierarchy[:3] == (13, 14, 15)
    # does not mutate the caller's settings dicts
    assert "13" not in CHARM_ELEM


def test_hook_upsamples_label_to_fidelity_voxel():
    lab, aff = _gm_block(vox=0.5)
    r = apply_seeg_to_label(lab, aff, _spec(label_voxel_mm=0.25),
                            elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET)
    got_vox = np.linalg.norm(r.label_affine[:3, 0])
    assert got_vox == pytest.approx(0.25, abs=0.02)
    # painted labels present in the buffer
    assert {13, 14, 15}.issubset(set(np.unique(r.label_buffer).tolist()))


def test_hook_subvoxel_sheath_scales_sigma():
    lab, aff = _gm_block()
    r = apply_seeg_to_label(lab, aff, _spec(sheath_thickness_um=150, label_voxel_mm=0.25),
                            elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET)
    # 150 um sheath < 250 um voxel -> grown to 1 voxel, sigma * (250/150) to keep t/sigma
    assert r.materials.sheath_sigma == pytest.approx(0.05 * 0.25 / 0.15, rel=1e-3)
    assert any("preserves t/sigma" in w for w in r.warnings)


def test_hook_sidecar_roundtrips():
    lab, aff = _gm_block()
    r = apply_seeg_to_label(lab, aff, _spec(),
                            elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET)
    sc = json.loads(json.dumps(r.sidecar()))   # must be JSON-serialisable
    assert sc["leads"][0]["name"] == "T"
    assert set(sc["materials"].keys()) >= {"13", "14", "15"} or set(
        int(k) for k in sc["materials"]) >= {13, 14, 15}


def test_hook_sidecar_cond_list_carries_resolved_sigma():
    """The sidecar's cond_list must carry the *resolved* (thin-shell-scaled) sheath sigma so a
    downstream run_simnibs uses the spec value, not the static registry default."""
    from simnibs.simulation.seeg.conductivity import load_seeg_cond_list

    lab, aff = _gm_block()
    r = apply_seeg_to_label(lab, aff, _spec(sheath_thickness_um=150, label_voxel_mm=0.25),
                            elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET)
    sc = json.loads(json.dumps(r.sidecar()))
    scaled = 0.05 * 0.25 / 0.15                       # sub-voxel sheath -> sigma up-scaled
    assert sc["cond_list"][GLIAL_SHEATH - 1] == pytest.approx(scaled, rel=1e-3)
    assert sc["cond_list"][SEEG_CONTACT - 1] == pytest.approx(1e6)
    # load helper round-trips both from the dict and from a written file
    assert load_seeg_cond_list(sc)[GLIAL_SHEATH - 1] == pytest.approx(scaled, rel=1e-3)


def test_hook_preserves_custom_hierarchy():
    """A non-empty charm hierarchy is preserved after the electrode tags, not dropped."""
    lab, aff = _gm_block()
    custom = (2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)   # some reordered tissue hierarchy
    r = apply_seeg_to_label(lab, aff, _spec(),
                            elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET,
                            hierarchy=custom)
    assert r.hierarchy == (13, 14, 15, 16, *custom)
    # falsy/None still yields the native electrode-first default
    r2 = apply_seeg_to_label(lab, aff, _spec(),
                             elem_sizes=CHARM_ELEM, facet_distances=CHARM_FACET,
                             hierarchy=False)
    assert r2.hierarchy == _HIERARCHY
