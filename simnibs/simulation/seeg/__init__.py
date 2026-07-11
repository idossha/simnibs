"""sEEG depth-electrode insertion for SimNIBS head models (conforming build).

Rebuild a subject head mesh with one or more sEEG depth leads embedded as **conforming**
subdomains -- a solid, isolated metal contact disc per macro on a constant-diameter,
continuous insulating body, wrapped in an optional glial/scar sheath -- so the standard
SimNIBS TDCS/TI solver consumes the result unchanged. Electrode geometry is chosen per lead
from a catalog (``SEEGLead(part_number=...)``); the sheath thickness and conductivity are
tunable. Built for temporal-interference field validation against intracranial recordings.

Quick start
-----------
>>> from simnibs.simulation.seeg import build_conforming_head, SEEGLead, CONF_FINE
>>> leads = [SEEGLead("R.HIPPO", "BF10R-SP21X-0C3",
...                   entry_mm=(81.52, 4.10, 5.40), target_mm=(29.86, 7.96, -1.42))]
>>> pl = build_conforming_head("m2m_ernie", leads, sheath_thickness_um=150, resolution=CONF_FINE)
>>> pl.mesh.write("ernie_seeg_conforming.msh")   # feed pl.mesh + pl.cond_list to run_simnibs

``preview_electrode`` builds a standalone 40 um smooth electrode (in a GM block) for
figures/QC. See ``README.md`` for the electrode model, presets, and citations.
"""

from __future__ import annotations

from ._params import (
    SEEG_CONTACT,
    SEEG_SHAFT,
    GLIAL_SHEATH,
    SEEG_TAGS,
    SEEGMaterials,
    SheathParams,
    ConformingResolution,
    CONF_STANDARD,
    CONF_FINE,
    CONF_ULTRA,
    MATERIALS_CHRONIC,
    MATERIALS_DENSE,
    MATERIALS_ACUTE,
    MATERIALS_PASSIVE,
)
from .catalog import ElectrodeCatalog, ElectrodeSpec
from .geometry import SEEGLead, perp_distance_and_axial
from .conductivity import ensure_seeg_registered, build_seeg_cond_list
from .embed import build_conforming_head, SEEGPlacement
from .charm_hook import apply_seeg_to_label, load_seeg_spec, SeegSpec
from .seeg_views import add_seeg_views, write_seeg_view_pos, SEEG_VIEW_SPEC
from .conforming import preview_electrode
from .sample import extract_axis_fields, AxisProfile
from .passive import passive_sample, PassiveProfile

__all__ = [
    # --- the build ---
    "build_conforming_head",
    "SEEGPlacement",
    "ConformingResolution",
    "CONF_STANDARD",
    "CONF_FINE",
    "CONF_ULTRA",
    "preview_electrode",                 # standalone smooth electrode (figures/QC)
    # --- native charm integration (charm --seeg) ---
    "apply_seeg_to_label",
    "load_seeg_spec",
    "SeegSpec",
    "add_seeg_views",
    "write_seeg_view_pos",
    "SEEG_VIEW_SPEC",
    # --- shared inputs ---
    "SEEGLead",
    "ElectrodeCatalog",
    "ElectrodeSpec",
    "SEEGMaterials",
    "SheathParams",
    "MATERIALS_CHRONIC",
    "MATERIALS_DENSE",
    "MATERIALS_ACUTE",
    "MATERIALS_PASSIVE",
    # --- tags / registration / conductivity ---
    "SEEG_CONTACT",
    "SEEG_SHAFT",
    "GLIAL_SHEATH",
    "SEEG_TAGS",
    "ensure_seeg_registered",
    "build_seeg_cond_list",
    # --- sampling / QC ---
    "extract_axis_fields",
    "AxisProfile",
    "passive_sample",
    "PassiveProfile",
    # --- geometry leaf ---
    "perp_distance_and_axial",
]
