"""Scientifically-informed defaults and tissue-tag constants for the sEEG module.

All numeric defaults are traceable to the literature synthesised for this module
(see ``README.md`` for the full citation list). The three new element tags used to
label the inserted electrode live in the free "tissue band" of the SimNIBS
``ElementTags`` enum (below ``ELECTRODE_RUBBER_START = 100``), so they:

* are treated as ordinary tissues by ``cond_utils.standard_cond()`` (tag <= ``TH_END``);
* survive the standard TI-envelope crop ``arange(TH_START, SALINE_START - 1)`` (< 499);
* do not collide with the reserved rubber (100-499) / saline (500-899) bands.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- New tissue tags (mirror the additions in utils/mesh_element_properties.py) ---
SEEG_CONTACT: int = 13   # metallic recording contact (Pt / Pt-Ir)
SEEG_SHAFT: int = 14     # insulating electrode body (polyurethane / silicone)
GLIAL_SHEATH: int = 15   # peri-electrode glial / encapsulation shell

SEEG_TAGS: tuple[int, int, int] = (SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH)

SEEG_TAG_NAMES: dict[int, str] = {
    SEEG_CONTACT: "SEEG_contact",
    SEEG_SHAFT: "SEEG_shaft",
    GLIAL_SHEATH: "Glial_sheath",
}

# Brain tissues (WM=1, GM=2, CSF=3). By default the encapsulation sheath is grown as a
# COMPLETE tube in ALL tissue the electrode passes (so no part of the metal/shaft ever
# contacts tissue directly -- see paint_leads_into_label). Pass ``sheath_tags=BRAIN_TAGS``
# to restrict the sheath to brain only (bare electrode in bone/scalp).
BRAIN_TAGS: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class SEEGMaterials:
    """Bulk conductivities (S/m) assigned to the three sEEG tissue tags.

    Defaults trade physical fidelity for FEM conditioning (see README §Materials):

    * ``contact_sigma`` -- a metal contact is ~10^7x more conductive than tissue and
      acts as a floating equipotential. Using the literal Pt value (9.4e6) creates a
      ~10^8 conductivity contrast that ill-conditions the stiffness matrix; 1e6 is
      already >=6 orders above tissue, so the recovered field is indistinguishable
      from the equipotential limit while the solve stays stable
      (Datta 2011; Lempka & McIntyre 2013; R1/R3).
    * ``shaft_sigma`` -- polyurethane is ~1e-12 S/m; 1e-5 is >=4 orders below tissue,
      which already blocks current, without the conditioning cost of a literal 1e-12.
    * ``sheath_sigma`` -- peri-electrode scar/encapsulation, a resistive shell *below* grey
      matter. Default 0.05 S/m = Karimi 2025's mouse-scar value at ~1 kHz (our 5-9 kHz
      carrier is in the same plateau; they report 0.019/0.05/0.28 S/m at 20 Hz/1 kHz/300 kHz).
      Sweep 0.05-0.16 via MATERIALS_* (Grill & Mortimer 1994; Butson 2006; Lempka 2013).
    """

    contact_sigma: float = 1.0e6
    shaft_sigma: float = 1.0e-5
    sheath_sigma: float = 0.05

    def as_tag_map(self) -> dict[int, float]:
        return {
            SEEG_CONTACT: self.contact_sigma,
            SEEG_SHAFT: self.shaft_sigma,
            GLIAL_SHEATH: self.sheath_sigma,
        }


# Convenient named variants for the sweep (R2 §0).
MATERIALS_CHRONIC = SEEGMaterials(sheath_sigma=0.10)   # subacute/chronic (default)
MATERIALS_DENSE = SEEGMaterials(sheath_sigma=0.05)     # densest fibrous capsule
MATERIALS_ACUTE = SEEGMaterials(sheath_sigma=1.0)      # acute edema / ECF-like shell
MATERIALS_PASSIVE = SEEGMaterials(                      # no metal perturbation (Huang-2017 baseline)
    contact_sigma=0.275, shaft_sigma=0.275, sheath_sigma=0.275
)


# Tissue tags the electrode BODY (contact + shaft) may replace when painting
# (any real tissue 1..12: scalp, bone, CSF, GM, WM). This is also the default set of
# tissues the sheath tube may grow in (full encapsulation along the whole track).
DISPLACE_TAGS: tuple[int, ...] = tuple(range(1, 13))


@dataclass(frozen=True)
class ConformingResolution:
    """Resolution knobs for the CONFORMING embed method (``build_conforming_head``).

    Geometric fidelity of the electrode is set by ``label_voxel_mm`` -- the voxel size the
    head tissue image is upsampled to before the electrodes are painted in (the CGAL surface
    hugs this voxel staircase to within ``electrode_facet_distance_mm``; RMS radial error
    ~= 0.5 * label_voxel_mm). A finer label is affordable because the mesh uses a RADIAL
    SIZING FIELD: small tets/facets only within ``reach_mm`` of the electrode axis
    (``electrode_edge_mm``), coarsening to ``bulk_edge_mm`` in the rest of the head -- so the
    whole-head tet count (and solve cost) stays small while the electrode is finely resolved.

    Sheath honesty: a 150 um sheath needs label_voxel <= ~75 um to be >=2 voxels thick; on a
    tractable whole-head grid it is ~1 voxel (thin but *consistent* with a constant-diameter
    body). For a fully-resolved sheath use ``preview_electrode`` (40 um standalone) or the
    surface-corefine route (staged).
    """

    label_voxel_mm: float = 0.25          # whole-head label voxel = geometric fidelity
    electrode_edge_mm: float = 0.12       # fine tet/facet size at the electrode
    electrode_facet_distance_mm: float = 0.03
    bulk_edge_mm: float = 2.5             # (image2mesh path) coarse tet far from the electrode
    reach_mm: float = 3.0                 # (image2mesh path) fine -> coarse ramp distance

    def to_create_mesh_kwargs(self) -> dict:
        """Native-charm per-tissue elem_sizes + facet_distances, plus the fine electrode.

        The tissue entries are charm's defaults (charm.ini [mesh]) so the head comes out at
        native SimNIBS quality; only the electrode tags 13/14/15 are added, finely sized.
        """
        el = dict(NATIVE_ELEM_SIZES)          # WM/GM/scalp/standard = native charm sizing
        fd = dict(NATIVE_FACET_DISTANCES)
        e = float(self.electrode_edge_mm)
        fdm = float(self.electrode_facet_distance_mm)
        for tag in ("13", "14", "15"):
            el[tag] = {"range": [e, e], "slope": 1.0}
            fd[tag] = {"range": [fdm, fdm], "slope": 1.0}
        return {"elem_sizes": el, "facet_distances": fd}


# Native charm mesh sizing (charm.ini / m2m settings.ini [mesh]) -- reproduces the standard
# SimNIBS head so build_conforming_head only ADDS the sEEG layer, not a coarser head.
NATIVE_ELEM_SIZES = {
    "standard": {"range": [1, 5], "slope": 1.0},
    "1": {"range": [1, 7], "slope": 1.0},        # WM
    "2": {"range": [1, 2], "slope": 1.0},        # GM (finest tissue)
    "5": {"range": [1, 10], "slope": 0.6},       # scalp
}
NATIVE_FACET_DISTANCES = {"standard": {"range": [0.1, 3], "slope": 0.5}}


# Presets: geometric-fidelity / cost ladder for the image2mesh + radial-sizing-field path
# (whole ernie head, crop_to_head). label_voxel_mm sets electrode fidelity (RMS ~ 0.5*vox).
CONF_STANDARD = ConformingResolution(0.30, 0.15, 0.04, 2.5, 3.0)  # ~0.15 mm RMS, fast
CONF_FINE = ConformingResolution(0.25, 0.12, 0.03, 2.5, 3.0)      # ~0.12 mm RMS (default)
CONF_ULTRA = ConformingResolution(0.20, 0.10, 0.025, 2.0, 3.0)    # ~0.10 mm RMS, heavier label


@dataclass(frozen=True)
class SheathParams:
    """Glial-sheath geometry knobs.

    ``thickness_um`` default 200 sits at the centre of Missey-2026 SI Fig S7's
    150-300 um sEEG estimate; sweep 100-500 um to bracket the DBS/FBR literature (R2 §1).
    """

    thickness_um: float = 150.0
    enabled: bool = True

    @property
    def thickness_mm(self) -> float:
        return self.thickness_um / 1000.0


# Descriptions registered alongside the tags (used by standard_cond()).
SEEG_TAG_DESCRIPTIONS: dict[int, str] = {
    SEEG_CONTACT: (
        "sEEG metallic recording contact (Pt / Pt-Ir); high-sigma equipotential "
        "limit 1e6 S/m for FEM stability (Datta 2011; Lempka 2013)"
    ),
    SEEG_SHAFT: (
        "sEEG insulating shaft (polyurethane/silicone); near-insulator 1e-5 S/m "
        "(physical ~1e-12; R1)"
    ),
    GLIAL_SHEATH: (
        "peri-electrode glial/encapsulation sheath; resistive shell 0.10 S/m, "
        "150-300 um (Grill&Mortimer 1994; Butson 2006; Yousif 2008; Missey 2026 SI S7)"
    ),
}
