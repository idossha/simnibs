"""Tests for the sEEG Gmsh view generator (charm --seeg .opt colouring).

Pure-numpy: a tiny duck-typed mesh exercises the .pos writer and the Visualization styling
without CGAL/gmsh. Locks the colours, the empty-structure skip, and the ColorTable-last
ordering that keeps ColormapAlpha from wiping the colour.
"""

from __future__ import annotations

import types

import numpy as np

from simnibs.simulation.seeg.seeg_views import (
    write_seeg_view_pos,
    add_seeg_views,
    SEEG_VIEW_SPEC,
)


def _fake_mesh():
    # triangles: 2x contact(1013), 1x shaft(1014), none sheath(1015); plus a tet (ignored)
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], float)
    nlist = np.array([[1, 2, 3, -1], [2, 3, 4, -1], [1, 2, 4, -1], [1, 2, 3, 4]])
    tag1 = np.array([1013, 1013, 1014, 13])
    etype = np.array([2, 2, 2, 4])                       # 2=triangle, 4=tet
    elm = types.SimpleNamespace(tag1=tag1, elm_type=etype, node_number_list=nlist)
    nodes = types.SimpleNamespace(node_coord=coords)
    return types.SimpleNamespace(elm=elm, nodes=nodes)


def test_write_pos_skips_empty_structures(tmp_path):
    fn = str(tmp_path / "v.pos")
    written = write_seeg_view_pos(_fake_mesh(), fn)
    assert [w[0] for w in written] == ["SEEG_contact", "SEEG_shaft"]  # sheath absent -> skipped
    txt = open(fn).read()
    assert txt.count("ST(") == 3                          # 2 contact + 1 shaft triangles
    assert 'View "SEEG_contact"' in txt
    assert 'View "Glial_sheath"' not in txt


def test_add_seeg_views_styles_and_merges(tmp_path):
    fn_mesh = str(tmp_path / "head.msh")
    vis = types.SimpleNamespace(merge=[], View=[])
    add_seeg_views(vis, _fake_mesh(), fn_mesh)
    assert vis.merge and vis.merge[0].endswith("head_seeg_views.pos")
    assert len(vis.View) == 2
    contact, shaft = vis.View
    assert contact.indx == 0 and shaft.indx == 1
    assert contact.ColorTable[0][:3] == [154, 154, 154]
    assert shaft.ColorTable[0][:3] == [71, 71, 71]
    assert contact.ColormapAlpha == 1.0
    # ColorTable must be the LAST emitted line (else ColormapAlpha wipes it -> red)
    lines = [l for l in str(contact).strip().split("\n") if l]
    assert lines[-1].startswith("View[0].ColorTable")
    assert str(contact).index("ColormapAlpha") < str(contact).index("ColorTable")


def test_add_seeg_views_indices_start_after_existing(tmp_path):
    fn_mesh = str(tmp_path / "head.msh")
    vis = types.SimpleNamespace(merge=[], View=["field0", "field1"])  # 2 pre-existing field views
    add_seeg_views(vis, _fake_mesh(), fn_mesh)
    assert [v.indx for v in vis.View[2:]] == [2, 3]        # sEEG views appended after fields


def test_seeg_view_spec_matches_targets():
    d = {t: (n, rgb, a) for t, n, rgb, a in SEEG_VIEW_SPEC}
    assert d[1013][1] == (154, 154, 154) and d[1013][2] == 1.0
    assert d[1014][1] == (71, 71, 71) and d[1014][2] == 1.0
    assert d[1015][1] == (162, 32, 242) and d[1015][2] == 0.5
