"""Gmsh visualization for the sEEG structures (contact / shaft / glial sheath).

When a head is built with ``charm --seeg`` the sEEG compartments (element tags 13-16,
surface tags 1013-1016) are just tissues in the mesh — Gmsh would draw them with an
arbitrary physical-group colour and no way to isolate them. This module adds a dedicated,
individually-toggleable **Gmsh View** per structure with a fixed colour, so the electrode
reads clearly against the tissue:

======================  ==================  =========
structure (surface tag)  colour              opacity
======================  ==================  =========
SEEG_contact (1013)      medium grey 154     opaque
SEEG_shaft   (1014)      dark grey   71       opaque
Glial_sheath (1015)      purple 162,32,242    50 %
Fibrous_sheath (1016)    tan 205,164,106      50 %
======================  ==================  =========

The colours are deliberately outside the "heat" field colormap so the electrode never
looks like part of an E-field. Each view is a lightweight scalar-triangle ``.pos`` written
next to the mesh and ``Merge``-d from the ``.opt`` (the same mechanism ``run_simnibs`` uses
for its electrode/scalp views), so the multi-GB head mesh itself is not bloated.

``add_seeg_views`` is called automatically from the charm mesh step when ``--seeg`` is
supplied (see ``segmentation/charm_main``); it is also usable on any solved result mesh
that carries tags 1013-1016.
"""

from __future__ import annotations

import os

import numpy as np

# (surface_tag, view_name, (R, G, B), alpha) -- names match the mesh physical groups.
# Colours are deliberately outside the E-field "heat" colormap; the two sheaths are visually
# distinct (glial = purple, fibrous = tan) so the segmented compartments read apart.
SEEG_VIEW_SPEC = [
    (1013, "SEEG_contact", (154, 154, 154), 1.0),
    (1014, "SEEG_shaft", (71, 71, 71), 1.0),
    (1015, "Glial_sheath", (162, 32, 242), 0.5),
    (1016, "Fibrous_sheath", (205, 164, 106), 0.5),
]

_TRIANGLE = 2  # Gmsh element type for a 3-node triangle


def write_seeg_view_pos(mesh, fn_pos) -> list[tuple[str, tuple[int, int, int], float]]:
    """Write a Gmsh ``.pos`` with one flat scalar-triangle view per sEEG structure.

    Each view contains only that structure's *surface* triangles (tag 1013-1016) with
    a constant scalar, so a solid ``ColorTable`` renders it as one flat colour. Structures
    absent from the mesh are skipped. Returns the ordered list of ``(name, rgb, alpha)``
    actually written (so the caller can style exactly those views, in order).
    """
    tag1 = mesh.elm.tag1
    etype = mesh.elm.elm_type
    node_list = mesh.elm.node_number_list
    coords = mesh.nodes.node_coord

    written: list[tuple[str, tuple[int, int, int], float]] = []
    with open(fn_pos, "w") as f:
        f.write("// sEEG structure views -- created by simnibs.simulation.seeg\n")
        for tag, name, rgb, alpha in SEEG_VIEW_SPEC:
            idx = np.where((tag1 == tag) & (etype == _TRIANGLE))[0]
            if idx.size == 0:
                continue
            tri_nodes = node_list[idx][:, :3]              # (N, 3) 1-based node ids
            xyz = coords[tri_nodes - 1].reshape(-1, 9)     # (N, 9) x0..z2
            f.write('View "%s" {\n' % name)
            np.savetxt(
                f, xyz,
                fmt="ST(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f){1,1,1};",
            )
            f.write("};\n")
            written.append((name, rgb, alpha))
    return written


def _solid_color_table(rgb: tuple[int, int, int], alpha: float = 1.0) -> list[list[int]]:
    """A 2-entry RGBA ColorTable of a single colour (flat fill for a constant field).

    The opacity is baked into the table's alpha channel. This is deliberate: Gmsh's
    ``ColormapAlpha`` only tints the colormap *when it is (re)generated*, and a direct
    ``ColorTable`` assignment (emitted last) overrides that regeneration -- so the sheath's
    50 % transparency has to live in the ColorTable itself, not in ColormapAlpha, to render.
    """
    a = int(round(255 * max(0.0, min(1.0, alpha))))
    entry = [int(rgb[0]), int(rgb[1]), int(rgb[2]), a]
    return [entry, list(entry)]


def add_seeg_views(vis, mesh, fn_mesh):
    """Add the sEEG structure views (contact, shaft, glial + fibrous sheath) to a ``gmsh_view.Visualization``.

    Writes ``<mesh>_seeg_views.pos`` next to ``fn_mesh``, registers it as a ``Merge`` in
    ``vis``, and appends one styled :class:`~simnibs.mesh_tools.gmsh_view.View` per structure
    (solid colour; sheath at 50 % opacity). View indices start after any views already in
    ``vis`` (so this is correct whether the base mesh has field views or none), and rely on
    the ColorTable-last ordering in ``View.__str__`` so ``ColormapAlpha`` does not wipe the
    colour. No-op (returns ``vis`` unchanged) if the mesh has no sEEG surface tags.
    """
    from simnibs.mesh_tools.gmsh_view import View

    fn_pos = os.path.splitext(fn_mesh)[0] + "_seeg_views.pos"
    written = write_seeg_view_pos(mesh, fn_pos)
    if not written:
        return vis

    vis.merge.append(fn_pos)
    try:
        len(vis.View)
    except TypeError:
        vis.View = []

    start = len(vis.View)
    for i, (name, rgb, alpha) in enumerate(written):
        vis.View.append(
            View(
                indx=start + i,
                Visible=1,
                ShowScale=0,
                RangeType=2,          # custom range so a constant field maps to the colour
                CustomMin=0,
                CustomMax=1,
                SaturateValues=1,
                # alpha is baked into the ColorTable (see _solid_color_table); ColormapAlpha
                # would be wiped by the ColorTable-last emission and is intentionally omitted.
                ColorTable=_solid_color_table(rgb, alpha),
            )
        )
    return vis
