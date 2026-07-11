"""Register the sEEG tissue tags and build conductivity lists for the FEM.

Two integration paths, both supported:

1. *Native* -- the three tags are added to ``utils/mesh_element_properties.py``
   (enum + tissue_* dicts). In a built SimNIBS this is picked up automatically by
   ``cond_utils.standard_cond()``.

2. *Runtime* -- ``ensure_seeg_registered()`` idempotently patches the same
   ``tissue_*`` module dicts at run time, so the module also works against a stock
   SimNIBS whose enum was not edited. This is a no-op when native registration is
   already present.
"""

from __future__ import annotations

from simnibs.utils import mesh_element_properties as _mep
from simnibs.utils import cond_utils as _cond_utils

from ._params import (
    SEEG_TAGS,
    SEEG_TAG_NAMES,
    SEEG_TAG_DESCRIPTIONS,
    SEEGMaterials,
)


def ensure_seeg_registered(materials: SEEGMaterials | None = None) -> None:
    """Ensure tags 13/14/15 exist in the central tissue registries.

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
