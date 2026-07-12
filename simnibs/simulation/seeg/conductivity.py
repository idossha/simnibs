"""Register the sEEG tissue tags and build conductivity lists for the FEM.

Two integration paths, both supported:

1. *Native* -- the four tags are added to ``utils/mesh_element_properties.py``
   (enum + tissue_* dicts). In a built SimNIBS this is picked up automatically by
   ``cond_utils.standard_cond()``.

2. *Runtime* -- ``ensure_seeg_registered()`` idempotently patches the same
   ``tissue_*`` module dicts at run time, so the module also works against a stock
   SimNIBS whose enum was not edited. This is a no-op when native registration is
   already present.
"""

from __future__ import annotations

import json
import os

from simnibs.utils import mesh_element_properties as _mep
from simnibs.utils import cond_utils as _cond_utils

from ._params import (
    SEEG_TAGS,
    SEEG_TAG_NAMES,
    SEEG_TAG_DESCRIPTIONS,
    SEEGMaterials,
)


def ensure_seeg_registered(materials: SEEGMaterials | None = None) -> None:
    """Ensure the sEEG tags 13-16 exist in the central tissue registries.

    Registers the *default* conductivities (or ``materials`` if given). Per-run
    overrides should go through :func:`build_seeg_cond_list` or by setting
    ``tdcs.cond[tag-1].value`` -- this function only guarantees the tags resolve.
    """
    materials = materials or SEEGMaterials()
    sigma = materials.as_tag_map()
    for tag in SEEG_TAGS:
        if tag not in _mep.tissue_tags:
            _mep.tissue_tags.append(tag)
        # Names/descriptions: keep any native text (setdefault, don't clobber).
        _mep.tissue_names.setdefault(tag, SEEG_TAG_NAMES[tag])
        _mep.tissue_conductivity_descriptions.setdefault(
            tag, SEEG_TAG_DESCRIPTIONS[tag]
        )
        # Conductivity: ALWAYS reflect the requested materials, so a sweep that calls
        # this with different SEEGMaterials updates standard_cond() (setdefault would
        # silently keep a pre-existing/native value and ignore the sweep).
        _mep.tissue_conductivities[tag] = sigma[tag]


def build_seeg_cond_list(
    base: list[float] | None = None,
    materials: SEEGMaterials | None = None,
) -> list[float]:
    """Return a per-tag conductivity list ready for ``cond_utils.cond2elmdata``.

    The list is indexed ``cond_list[tag - 1]`` (SimNIBS convention). It is built
    from ``standard_cond()`` and then the three sEEG indices are patched with
    ``materials`` (overriding whatever default was registered), so a sweep over
    sheath conductivity needs no library edit.
    """
    materials = materials or SEEGMaterials()
    ensure_seeg_registered(materials)
    if base is None:
        base = [c.value if c.value is not None else 0.0 for c in _cond_utils.standard_cond()]
    else:
        base = list(base)
    need = max(SEEG_TAGS)
    if len(base) < need:
        base = base + [0.0] * (need - len(base))
    for tag, value in materials.as_tag_map().items():
        base[tag - 1] = float(value)
    return base


def load_seeg_cond_list(sidecar, materials: "SEEGMaterials | None" = None) -> list[float]:
    """Return the ``cond_list[tag-1]`` vector to solve a ``charm --seeg`` mesh with.

    This is where the sEEG conductivity is set -- at *simulation* time (charm itself only
    segments + meshes and assigns the default sigma). ``sidecar`` is the ``*_seeg.json`` path
    (or loaded dict) written next to the mesh::

        from simnibs.simulation.seeg import load_seeg_cond_list
        cl = load_seeg_cond_list("ernie_seeg.json")          # charm's default sigma
        for i, v in enumerate(cl):
            tdcs.cond[i].value = v                           # tags 13-16 carry sigma

    Pass ``materials`` to solve with a *different* sheath sigma (e.g. a sweep) -- no scaling is
    applied; the sheath is meshed at true thickness, so the sigma is used as given::

        cl = load_seeg_cond_list("ernie_seeg.json",
                                 materials=SEEGMaterials(fibrous_sheath_sigma=0.30))

    With ``materials=None`` and an explicit ``cond_list`` in the sidecar, that (default) vector
    is returned verbatim; otherwise it is rebuilt from the recorded ``materials`` map.
    """
    if isinstance(sidecar, (str, os.PathLike)):
        with open(sidecar) as fh:
            sidecar = json.load(fh)
    if not isinstance(sidecar, dict):
        raise TypeError(f"sidecar must be a path or dict, got {type(sidecar).__name__}")

    if materials is not None:
        return build_seeg_cond_list(materials=materials)

    cl = sidecar.get("cond_list")
    if cl is not None:
        return [float(v) for v in cl]

    mats = sidecar.get("materials") or {}
    recorded = SEEGMaterials(
        contact_sigma=float(mats.get("13", SEEGMaterials.contact_sigma)),
        shaft_sigma=float(mats.get("14", SEEGMaterials.shaft_sigma)),
        sheath_sigma=float(mats.get("15", SEEGMaterials.sheath_sigma)),
        fibrous_sheath_sigma=float(mats.get("16", SEEGMaterials.fibrous_sheath_sigma)),
    )
    return build_seeg_cond_list(materials=recorded)
