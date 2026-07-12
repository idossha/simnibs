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


def load_seeg_cond_list(sidecar) -> list[float]:
    """Return the ``cond_list[tag-1]`` conductivity vector recorded by ``charm --seeg``.

    ``sidecar`` is the ``*_seeg.json`` path (or the already-loaded dict) written next to the
    mesh. The returned list carries the *resolved* sEEG conductivities -- including any
    thin-shell sigma up-scaling applied when the sheath was sub-voxel -- so a downstream solve
    uses the spec's sheath conductivity rather than the static registry default::

        from simnibs.simulation.seeg import load_seeg_cond_list
        S = sim_struct.SESSION(); tdcs = S.add_tdcslist()
        cl = load_seeg_cond_list("ernie_seeg.json")
        for i, v in enumerate(cl):
            tdcs.cond[i].value = v            # tags 13-16 now carry the resolved sigma

    Falls back to rebuilding the list from the sidecar's ``materials`` map if an older sidecar
    without an explicit ``cond_list`` is passed.
    """
    if isinstance(sidecar, (str, os.PathLike)):
        with open(sidecar) as fh:
            sidecar = json.load(fh)
    if not isinstance(sidecar, dict):
        raise TypeError(f"sidecar must be a path or dict, got {type(sidecar).__name__}")

    cl = sidecar.get("cond_list")
    if cl is not None:
        return [float(v) for v in cl]

    mats = sidecar.get("materials") or {}
    materials = SEEGMaterials(
        contact_sigma=float(mats.get("13", SEEGMaterials.contact_sigma)),
        shaft_sigma=float(mats.get("14", SEEGMaterials.shaft_sigma)),
        sheath_sigma=float(mats.get("15", SEEGMaterials.sheath_sigma)),
        fibrous_sheath_sigma=float(mats.get("16", SEEGMaterials.fibrous_sheath_sigma)),
    )
    return build_seeg_cond_list(materials=materials)
