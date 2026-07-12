"""Tests for the sEEG module shared inputs (catalog, geometry, conductivity).

Pure-numpy tests run anywhere; the conductivity-registration test needs SimNIBS utils.
"""

from __future__ import annotations

import numpy as np
import pytest

from simnibs.simulation.seeg.catalog import ElectrodeCatalog, ElectrodeSpec
from simnibs.simulation.seeg.geometry import SEEGLead, perp_distance_and_axial
from simnibs.simulation.seeg._params import (
    SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH, FIBROUS_SHEATH, SEEGMaterials,
)


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def test_catalog_default_loads_study_part():
    cat = ElectrodeCatalog.default()
    assert "BF10R-SP21X-0C3" in cat
    spec = cat["BF10R-SP21X-0C3"]
    assert spec.n_contacts == 10
    assert spec.contact_len_mm == pytest.approx(1.57)
    assert spec.spacing_mm == pytest.approx(5.0)
    assert spec.contact_radius_mm == pytest.approx(0.64)


def test_catalog_unknown_part_raises():
    with pytest.raises(KeyError):
        ElectrodeCatalog.default()["NOT-A-PART"]


def test_electrodespec_rejects_nonpositive():
    with pytest.raises(ValueError):
        ElectrodeSpec("X", "m", "f", n_contacts=0, contact_len_mm=1,
                      contact_dia_mm=1, shaft_dia_mm=1, spacing_mm=5)


def test_catalog_missing_required_field_names_part(tmp_path):
    import json
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({
        "MYLAB-01": {"manufacturer": "Lab", "n_contacts": 8, "contact_len_mm": 2.0,
                     "contact_dia_mm": 0.8, "shaft_dia_mm": 0.8, "spacing_mm": 3.5}
    }))
    with pytest.raises(ValueError, match="MYLAB-01"):
        ElectrodeCatalog(str(p))


def test_body_radius_constant_defaults_to_contact():
    spec = ElectrodeCatalog.default()["BF10R-SP21X-0C3"]
    # a real electrode is a constant-Ø body; default body radius = contact radius
    assert spec.body_radius_mm == pytest.approx(spec.contact_radius_mm)
    # explicit body_dia overrides
    s2 = ElectrodeSpec("X", "m", "f", 8, 2.0, 1.1, 1.3, 5.0, body_dia_mm=1.4)
    assert s2.body_radius_mm == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_axis_unit_and_length():
    lead = SEEGLead("L", "P", entry_mm=(0, 0, 0), target_mm=(10, 0, 0))
    assert lead.length_mm == pytest.approx(10.0)
    np.testing.assert_allclose(lead.axis_unit(), [1, 0, 0])


def test_contact_centers_march_from_tip():
    spec = ElectrodeCatalog.default()["BF10R-SP21X-0C3"]
    lead = SEEGLead("L", "BF10R-SP21X-0C3", entry_mm=(0, 0, 0), target_mm=(60, 0, 0),
                    first_contact_depth_mm=2.0)
    c = lead.contact_centers(spec)
    assert c.shape == (10, 3)
    assert c[0, 0] == pytest.approx(58.0)          # 2 mm proximal to tip
    assert c[1, 0] == pytest.approx(53.0)          # 5 mm pitch toward entry
    assert c[-1, 0] == pytest.approx(58.0 - 9 * 5.0)


def test_contact_centers_overflow_raises():
    spec = ElectrodeCatalog.default()["BF10R-SP21X-0C3"]
    lead = SEEGLead("L", "BF10R-SP21X-0C3", entry_mm=(0, 0, 0), target_mm=(40, 0, 0))
    with pytest.raises(ValueError):
        lead.contact_centers(spec)


def test_contact_centers_first_depth_past_tip_raises():
    spec = ElectrodeCatalog.default()["BF10R-SP21X-0C3"]
    lead = SEEGLead("L", "BF10R-SP21X-0C3", entry_mm=(0, 0, 0), target_mm=(60, 0, 0),
                    first_contact_depth_mm=0.3)
    with pytest.raises(ValueError):
        lead.contact_centers(spec)


def test_perp_distance_and_axial():
    A = np.array([0.0, 0, 0]); B = np.array([10.0, 0, 0])
    pts = np.array([[5.0, 1.0, 0.0], [5.0, 0.0, 2.0], [-3.0, 0.0, 0.0]])
    dist, axial = perp_distance_and_axial(pts, A, B)
    assert dist[0] == pytest.approx(1.0)
    assert dist[1] == pytest.approx(2.0)
    assert axial[2] == pytest.approx(0.0)          # clamps behind A
    assert dist[2] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Conductivity registration (needs simnibs.utils)
# --------------------------------------------------------------------------- #
def test_conductivity_registration_and_cond_list():
    pytest.importorskip("simnibs.utils.cond_utils")
    from simnibs.simulation.seeg.conductivity import (
        ensure_seeg_registered, build_seeg_cond_list,
    )
    from simnibs.utils import cond_utils

    ensure_seeg_registered(SEEGMaterials(contact_sigma=1e6, shaft_sigma=1e-5, sheath_sigma=0.05))
    S = cond_utils.standard_cond()
    assert S[SEEG_CONTACT - 1].value == pytest.approx(1e6)
    assert S[SEEG_SHAFT - 1].value == pytest.approx(1e-5)

    cl = build_seeg_cond_list(materials=SEEGMaterials(sheath_sigma=0.25))
    assert cl[GLIAL_SHEATH - 1] == pytest.approx(0.25)
    assert cl[SEEG_CONTACT - 1] == pytest.approx(1e6)
    assert cl[2 - 1] == pytest.approx(0.275, rel=1e-3)   # GM untouched


def test_registration_reflects_sweep_materials():
    pytest.importorskip("simnibs.utils.cond_utils")
    from simnibs.simulation.seeg.conductivity import ensure_seeg_registered
    from simnibs.utils import cond_utils, mesh_element_properties as mep

    ensure_seeg_registered(SEEGMaterials(sheath_sigma=0.05))
    assert cond_utils.standard_cond()[GLIAL_SHEATH - 1].value == pytest.approx(0.05)
    ensure_seeg_registered(SEEGMaterials(sheath_sigma=0.30))
    assert mep.tissue_conductivities[GLIAL_SHEATH] == pytest.approx(0.30)


# --------------------------------------------------------------------------- #
# Single-source-of-truth guards (fail loudly on silent drift)
# --------------------------------------------------------------------------- #
def test_native_sheath_sigma_matches_module_default():
    """The native registry default (what a stock run_simnibs uses) must equal the module
    default so 'charm --seeg' then 'run_simnibs' does not silently solve a different sigma.

    Read the committed source value (not the live dict, which ensure_seeg_registered mutates)."""
    import ast, inspect
    from simnibs.utils import mesh_element_properties as mep
    tree = ast.parse(inspect.getsource(mep))

    def _targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AnnAssign):        # tissue_conductivities: dict[int,float] = {...}
            return [node.target]
        return []

    src_sheath = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
            getattr(t, "id", None) == "tissue_conductivities" for t in _targets(node)
        ):
            if not isinstance(node.value, ast.Dict):
                continue
            for key, val in zip(node.value.keys, node.value.values):
                if getattr(key, "attr", None) == "GLIAL_SHEATH":   # ElementTags.GLIAL_SHEATH
                    src_sheath = ast.literal_eval(val)
    assert src_sheath is not None, "GLIAL_SHEATH not found in tissue_conductivities source"
    assert src_sheath == pytest.approx(SEEGMaterials().sheath_sigma)


def test_segmented_materials_carry_fibrous_tag():
    """The fibrous sheath (tag 16) is a first-class material: in as_tag_map, scaled by
    with_scaled_sheath alongside the glial sheath, and present in the solver cond_list."""
    from simnibs.simulation.seeg.conductivity import build_seeg_cond_list
    m = SEEGMaterials()
    tm = m.as_tag_map()
    assert tm[GLIAL_SHEATH] == pytest.approx(0.05)
    assert tm[FIBROUS_SHEATH] == pytest.approx(0.16)
    # sub-voxel scaling scales BOTH sheaths, leaves contact/shaft untouched
    s = m.with_scaled_sheath(2.0)
    assert s.sheath_sigma == pytest.approx(0.10)
    assert s.fibrous_sheath_sigma == pytest.approx(0.32)
    assert s.contact_sigma == m.contact_sigma and s.shaft_sigma == m.shaft_sigma
    # cond_list carries the fibrous value at index tag-1
    cl = build_seeg_cond_list(materials=SEEGMaterials(fibrous_sheath_sigma=0.2))
    assert cl[FIBROUS_SHEATH - 1] == pytest.approx(0.2)
    assert cl[GLIAL_SHEATH - 1] == pytest.approx(0.05)


def test_seeg_hierarchy_tail_matches_meshing_default():
    """SEEG_HIERARCHY must be the sEEG tags followed by meshing.create_mesh's own hierarchy=None
    default -- if upstream changes that default, this fails instead of silently reordering."""
    from simnibs.simulation.seeg._params import SEEG_HIERARCHY, SEEG_TAGS
    from simnibs.mesh_tools import meshing
    import inspect
    src = inspect.getsource(meshing.create_mesh)
    # the native default is the tuple literal assigned when hierarchy is None
    assert "(1, 2, 9, 3, 4, 8, 7, 6, 11, 10, 12, 5)" in src
    native = (1, 2, 9, 3, 4, 8, 7, 6, 11, 10, 12, 5)
    assert SEEG_HIERARCHY == (*SEEG_TAGS, *native)
