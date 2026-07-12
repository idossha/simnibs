"""Scientifically-informed defaults and tissue-tag constants for the sEEG module.

All numeric defaults are traceable to the literature synthesised for this module
(see ``README.md`` for the full citation list). The four new element tags used to
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
GLIAL_SHEATH: int = 15   # peri-electrode glial / encapsulation shell (brain GM/WM)
FIBROUS_SHEATH: int = 16 # peri-electrode fibrous / granulation tract (bone / scalp / soft tissue)

SEEG_TAGS: tuple[int, ...] = (SEEG_CONTACT, SEEG_SHAFT, GLIAL_SHEATH, FIBROUS_SHEATH)
SHEATH_TAGS: tuple[int, int] = (GLIAL_SHEATH, FIBROUS_SHEATH)

# Element-tag meshing priority: electrode tags FIRST so ``reconstruct_unique_surface`` keeps
# the electrode/tissue interface (clean 1013-1016 shells), then the native create_mesh
# tissue order. This is exactly ``meshing.create_mesh``'s ``hierarchy=None`` default with the
# four sEEG tags prepended -- single source of truth (both embed.py and charm_hook.py import
# it); ``tests/test_seeg.py`` asserts the tail still matches the upstream default so an upstream
# change fails the test instead of silently altering meshes.
_NATIVE_MESH_HIERARCHY: tuple[int, ...] = (1, 2, 9, 3, 4, 8, 7, 6, 11, 10, 12, 5)
SEEG_HIERARCHY: tuple[int, ...] = SEEG_TAGS + _NATIVE_MESH_HIERARCHY

SEEG_TAG_NAMES: dict[int, str] = {
    SEEG_CONTACT: "SEEG_contact",
    SEEG_SHAFT: "SEEG_shaft",
    GLIAL_SHEATH: "Glial_sheath",
    FIBROUS_SHEATH: "Fibrous_sheath",
}

# Brain tissue tags (WM=1, GM=2, CSF=3). The sheath is host-keyed (see SHEATH_TAG_BY_HOST /
# paint_leads_into_label): brain GM/WM -> glial, bone/scalp/soft -> fibrous, CSF/blood -> bare.
# BRAIN_TAGS is kept as a convenience constant for callers that want to reason about brain.
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
    * ``sheath_sigma`` -- peri-electrode glial scar/encapsulation, a resistive shell *below*
      grey matter. The measured source is **Evers et al. 2022** (rat chronic-DBS impedance
      spectroscopy; this is Karimi 2025's ref [48], NOT a Karimi mouse measurement):
      0.019 / 0.05 / 0.28 S/m at 20 Hz / 1 kHz / 300 kHz -- i.e. POSITIVE dispersion, not a
      plateau. The default 0.05 is the *1 kHz* value; log-interpolated to the 5-9 kHz carrier
      the chronic value is ~**0.08-0.10 S/m** (also the DBS-FEM consensus: Butson 2006 0.1,
      Yousif 2008 0.125, Alonso & Wardell 2015 0.1). Sweep 0.05-0.20; note sEEG monitoring is
      SUBACUTE (~1-2 wk) so resolving edema may push it higher (0.15-0.5, up to acute ~1.7).
      This value is a glial-scar model valid only in BRAIN (GM/WM). The bone/scalp/soft-tissue
      tract hosts a different reactive tissue (fibrous/granulation, ``fibrous_sheath_sigma``,
      tag 16), and CSF hosts none -- see ``SHEATH_TAG_BY_HOST`` and README §6.7. The contrast
      with the host even flips sign (resistive in GM, a conductive shunt in bone).
    """

    contact_sigma: float = 1.0e6
    shaft_sigma: float = 1.0e-5
    sheath_sigma: float = 0.05            # GLIAL sheath (brain GM/WM)
    fibrous_sheath_sigma: float = 0.16    # FIBROUS sheath (bone/scalp/soft-tissue tract)

    def as_tag_map(self) -> dict[int, float]:
        return {
            SEEG_CONTACT: self.contact_sigma,
            SEEG_SHAFT: self.shaft_sigma,
            GLIAL_SHEATH: self.sheath_sigma,
            FIBROUS_SHEATH: self.fibrous_sheath_sigma,
        }

    def with_scaled_sheath(self, scale: float) -> "SEEGMaterials":
        """Copy with BOTH sheath conductivities multiplied by ``scale``.

        Used when a sub-voxel sheath is meshed thicker than physical: scaling sigma by
        (t_mesh / t_phys) preserves each shell's sheet resistance t/sigma (thin-shell
        equivalence). Contact/shaft are unchanged.
        """
        return SEEGMaterials(
            contact_sigma=self.contact_sigma,
            shaft_sigma=self.shaft_sigma,
            sheath_sigma=self.sheath_sigma * scale,
            fibrous_sheath_sigma=self.fibrous_sheath_sigma * scale,
        )


# Convenient named variants for the sweep (R2 §0). NOTE: the module default (SEEGMaterials(),
# README §6.2) is 0.05 S/m == MATERIALS_DENSE; MATERIALS_CHRONIC (0.10) is the chronic-scar
# literature centre and a natural sweep target, not the default.
MATERIALS_DENSE = SEEGMaterials(sheath_sigma=0.05)     # dense fibrous capsule == SEEGMaterials() default
MATERIALS_CHRONIC = SEEGMaterials(sheath_sigma=0.10)   # subacute/chronic scar (literature centre; sweep target)
MATERIALS_ACUTE = SEEGMaterials(sheath_sigma=1.0)      # acute edema / ECF-like shell
MATERIALS_PASSIVE = SEEGMaterials(                      # no metal perturbation (Huang-2017 baseline)
    contact_sigma=0.275, shaft_sigma=0.275, sheath_sigma=0.275
)


# Tissue tags the electrode BODY (contact + shaft) may replace when painting (any real
# tissue 1..12: scalp, bone, CSF, GM, WM). The sheath, by contrast, is grown per
# SHEATH_TAG_BY_HOST (host-keyed segmented sheath), not over all of these.
DISPLACE_TAGS: tuple[int, ...] = tuple(range(1, 13))


# --------------------------------------------------------------------------------------- #
# Segmented sheath: which reactive tissue forms in which host (the ONLY sheath model)
# --------------------------------------------------------------------------------------- #
# The reactive tissue around a chronic depth electrode is NOT one material: it differs by the
# host tissue the electrode passes, and its contrast with the host even FLIPS SIGN. Each sheath
# voxel is therefore tagged by the host it grows in -- GLIAL_SHEATH (15) in brain, FIBROUS_SHEATH
# (16) in the bone/scalp/soft-tissue tract -- and the two tags carry different conductivities
# (SEEGMaterials.sheath_sigma 0.05 vs fibrous_sheath_sigma 0.16). Hosts NOT in this map get NO
# sheath: the shaft is left bare there (correct for CSF -- a CSF-bathed shaft has no encapsulation
# -- and for blood/eyes/cartilage). Sources: 2026-07-11 literature synthesis (Evers 2022;
# Grill & Mortimer 1994; Butson 2006; Yousif 2008; Alonso & Wardell 2015; Karimi 2025; IT'IS DB) --
# see README §6.7 "Segmented sheath".
#
#   host tissue (tag, sigma)    reactive tissue          -> sheath tag   contrast vs host
#   WM (1, 0.126), GM (2, 0.275) glial + fibrous scar    -> 15 GLIAL     resistive in GM / ~iso in WM
#   CSF (3, 1.654)              none (free shaft)         -> (bare)       a shell here is spurious
#   bone 4/7/8 (0.008-0.025)    fibrous/granulation tract -> 16 FIBROUS   10-60x CONDUCTIVE SHUNT
#   scalp 5, muscle 10, fat 12  fibrous exit-tract scar   -> 16 FIBROUS   mild, far from contacts
#   blood 9, eyes 6, cart 11    none                      -> (bare)       n/a
SHEATH_TAG_BY_HOST: dict[int, int] = {
    1: GLIAL_SHEATH,     # WM
    2: GLIAL_SHEATH,     # GM
    4: FIBROUS_SHEATH,   # bone (generic)
    5: FIBROUS_SHEATH,   # scalp
    7: FIBROUS_SHEATH,   # compact bone
    8: FIBROUS_SHEATH,   # spongy bone
    10: FIBROUS_SHEATH,  # muscle
    12: FIBROUS_SHEATH,  # fat
    # CSF(3), eyes(6), blood(9), cartilage(11) intentionally absent -> bare shaft (no sheath)
}


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
        native SimNIBS quality; only the electrode tags 13-16 are added, finely sized.
        """
        el = dict(NATIVE_ELEM_SIZES)          # WM/GM/scalp/standard = native charm sizing
        fd = dict(NATIVE_FACET_DISTANCES)
        augment_electrode_sizes(
            el, fd, float(self.electrode_edge_mm), float(self.electrode_facet_distance_mm)
        )
        return {"elem_sizes": el, "facet_distances": fd}


def augment_electrode_sizes(
    elem_sizes: dict, facet_distances: dict, edge_mm: float, facet_mm: float
) -> None:
    """In-place: add the fine electrode elem_size / facet_distance for the sEEG tags (13-16).

    Single source for the electrode-sizing block shared by ``ConformingResolution.
    to_create_mesh_kwargs`` (build_conforming_head) and ``charm_hook.apply_seeg_to_label``,
    so the two paths cannot drift. The tissue entries the caller already holds are untouched.
    """
    for tag in ("13", "14", "15", "16"):
        elem_sizes[tag] = {"range": [edge_mm, edge_mm], "slope": 1.0}
        facet_distances[tag] = {"range": [facet_mm, facet_mm], "slope": 1.0}


def sheath_paint_warnings(counts: dict[int, int]) -> list[str]:
    """Warnings for empty sEEG tags after painting/meshing.

    Warns if the electrode CORE (contact 13 / shaft 14) is empty, or if NO sheath (15+16)
    formed at all. Deliberately does NOT warn when only one sheath *type* is absent: a purely
    intracerebral track legitimately has no fibrous sheath (16), and a track that never enters
    brain has no glial sheath (15) -- so per-tag warnings on 15/16 would fire on normal runs.
    """
    w: list[str] = []
    if counts.get(SEEG_CONTACT, 0) == 0:
        w.append("SEEG contact (13) got 0 voxels (electrode outside tissue?)")
    if counts.get(SEEG_SHAFT, 0) == 0:
        w.append("SEEG shaft (14) got 0 voxels (electrode outside tissue?)")
    if counts.get(GLIAL_SHEATH, 0) + counts.get(FIBROUS_SHEATH, 0) == 0:
        w.append("no sheath (15/16) formed (thinner than a voxel, or track only in unmapped "
                 "hosts such as CSF)")
    return w


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
        "peri-electrode glial/encapsulation sheath (brain GM/WM); resistive shell 0.05 S/m "
        "@1kHz (~0.10 at 5-9 kHz; sweep 0.05-0.20), 150-300 um (Evers 2022; Grill&Mortimer "
        "1994; Butson 2006; Yousif 2008; Missey 2026 SI S7)"
    ),
    FIBROUS_SHEATH: (
        "peri-electrode fibrous/granulation tract tissue (bone/scalp/soft-tissue segments); "
        "0.16 S/m ~frequency-independent (Grill&Mortimer 1994); a conductive shunt vs bone"
    ),
}
