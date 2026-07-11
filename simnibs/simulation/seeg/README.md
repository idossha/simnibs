# `simnibs.simulation.seeg` — sEEG depth electrodes + glial sheath in SimNIBS

Insert one or more **sEEG depth leads** — metallic recording contacts + an insulating
shaft, fully wrapped in an adjustable **glial/encapsulation sheath** — into a SimNIBS head
model as a **native, optional step of the `charm` pipeline**. The electrode is built from a
scientifically-sourced catalog keyed by manufacturer part number; the sheath thickness and
conductivity are the tunable knobs. Built for temporal-interference (TI) field validation
against intracranial recordings, reproducing the near-contact field enhancement reported in
Missey et al. 2026 (SI Fig S7).

This is a single-file reference for the module: what it does, how to use it, the electrode
model, and — the heart of it — a **table of every decision variable with its value and the
literature it came from** (§6). Cited PDFs are collected in [`references/`](references/).

---

## 1. Why model the electrode at all?

A metallic contact (~10⁶–10⁷× more conductive than tissue) surrounded by a **resistive
glial sheath** locally reshapes the field. Missey-2026 SI Fig S7 finds the TI modulation is
enhanced **2–7.2×** near sEEG contacts but decays to background **within ~400 µm** — i.e.
within the typical **150–300 µm glial sheath**. Whether that perturbation matters for
validation depends entirely on the sheath, so it is a first-class, adjustable parameter.

The electrode is modelled the way the intracranial-implant FEM literature models it
(Karimi 2025; Lempka 2013; Datta 2011): **solid, isolated metal contact discs** (an
equipotential/PEC limit) on a **continuous constant-diameter insulating body**, with **no
conductive core** (each contact is individually wired; the leads are mutually isolated).
Two modelling requirements specific to this study are enforced by construction:

1. **Full length, skin → tip.** The shaft is painted along the entire entry→tip trajectory
   through scalp → bone → CSF → GM → WM (the entry is extended outward so the rod reliably
   reaches the outer skin).
2. **Complete encapsulation — no bare GM/WM.** The sheath is a gap-free tube built by
   morphological dilation of the electrode body, so **no metal or shaft voxel is ever
   face-adjacent to bare GM/WM** (or any tissue) — the scar always sits between electrode
   and parenchyma. By default the tube wraps the whole track (all tissue); pass
   `sheath_tags=(1,2,3)` to restrict it to brain.

---

## 2. How to build a head with electrodes

The electrode compartments are **off by default**; supplying electrode locations turns them
on. Electrode *type* is chosen per lead by `part_number` against the catalog (§5).

### A) Recommended — native `charm --seeg` (fully integrated)

```bash
# builds the head exactly as native SimNIBS, then paints in the sEEG compartments
charm --mesh ernie --seeg leads.json           # re-mesh an existing m2m_ernie
# or as part of a fresh run:  charm ernie T1.nii.gz [T2.nii.gz] --seeg leads.json
```

`leads.json`:

```json
{
  "leads": [
    {"name": "R.HIPPO", "part_number": "BF10R-SP21X-0C3",
     "entry_mm": [81.52, 4.10, 5.40], "target_mm": [29.86, 7.96, -1.42],
     "n_contacts": 10, "first_contact_depth_mm": 2.0}
  ],
  "sheath_thickness_um": 150,
  "label_voxel_mm": 0.25,
  "electrode_edge_mm": 0.12,
  "electrode_facet_distance_mm": 0.03,
  "materials": {"contact_sigma": 1e6, "shaft_sigma": 1e-5, "sheath_sigma": 0.05}
}
```

Only `leads` is required; everything else falls back to the study defaults in §6. charm
writes the usual `ernie.msh` (now carrying tags 13/14/15) plus an `ernie_seeg.json`
provenance sidecar and an `ernie_seeg_views.pos` companion. The `.opt` automatically gives
each sEEG structure its own toggleable, colour-coded **Gmsh view** — `SEEG_contact` medium
grey, `SEEG_shaft` dark grey, `Glial_sheath` purple at 50 % opacity (deliberately outside
the E-field "heat" colormap). The views are lightweight scalar-triangle surfaces `Merge`-d
from the `.opt`, so the multi-GB mesh is not bloated. The tissue mesh uses charm's own
`elem_sizes`, so the head is **native SimNIBS quality**; only the electrode is finely sized. The label is locally upsampled to
`label_voxel_mm` for a smooth rod (RMS radial error ≈ ½·voxel). This makes charm's mesh step
heavier (voxel-bound preprocessing: whole head at 0.25 mm ≈ 90–120 min).

### B) Programmatic — `build_conforming_head`

Same result without the charm CLI (takes an `m2m` folder or a `(label, affine)` tuple):

```python
from simnibs.simulation.seeg import build_conforming_head, SEEGLead, CONF_FINE

lead = SEEGLead("R.HIPPO", "BF10R-SP21X-0C3",
                entry_mm=(81.52, 4.10, 5.40), target_mm=(29.86, 7.96, -1.42))
pl = build_conforming_head("m2m_ernie", [lead], sheath_thickness_um=150, resolution=CONF_FINE)
pl.mesh.write("ernie_seeg.msh")     # feed pl.mesh + pl.cond_list to run_simnibs
```

`preview_electrode()` builds a *standalone* smooth electrode in a GM preview block for
figures/QC (not embedded in a head). Downstream field sampling:

```bash
simnibs_python -m simnibs.simulation.seeg fields \
    --result-mesh TI.msh --leads leads.json --out axis_profiles.csv
```

---

## 3. Architecture (one-directional layers)

The module is a thin, self-contained package under `simnibs/simulation/seeg/`. Only the
`charm --seeg` hook touches the rest of SimNIBS; everything else is leaf code.

| File | Responsibility |
|------|----------------|
| `_params.py` | tags (13/14/15), `SEEGMaterials`, `ConformingResolution`/`CONF_*`, native charm sizing |
| `catalog.py` | `ElectrodeSpec`/`ElectrodeCatalog` + bundled `seeg_catalog.json` (BF/RD/SD/DIXI/PMT) |
| `geometry.py` | `SEEGLead` (entry→tip trajectory) + `perp_distance_and_axial` (pure numpy) |
| `_raster.py` | rasterise leads into the tissue label: full-length body + complete sheath tube |
| `conductivity.py` | `ensure_seeg_registered` / `build_seeg_cond_list` (feeds the stock solver) |
| `embed.py` | `build_conforming_head` — paint + `create_mesh` → tagged head `SEEGPlacement` |
| `charm_hook.py` | `apply_seeg_to_label` — the single native hook charm's mesh step calls |
| `conforming.py` | `preview_electrode` — standalone smooth reference electrode (figures/QC) |
| `sample.py`, `passive.py` | extract carrier/AM E-field along each electrode axis from a solved mesh |
| `__main__.py` | CLI (`conforming`, `fields`) |

The three new element tags live in the free tissue band of `utils/mesh_element_properties.py`
(`SEEG_CONTACT=13`, `SEEG_SHAFT=14`, `GLIAL_SHEATH=15`), so `cond_utils.standard_cond()` and
the standard TDCS/TI solver treat them as ordinary tissues — **no solver changes**. The module
also self-registers them at import (`ensure_seeg_registered`) so it works against a stock
SimNIBS install too.

---

## 4. Electrode geometry (Ad-Tech, study electrode)

Model each **macro contact** as a cylindrical ring; the insulating body shares the contact
outer diameter (constant-Ø rod, contacts differ only axially → smooth rod + constant sheath
annulus). Numbers are the Ad-Tech 2015 catalogue's own, each cross-checked against an
external source (distributor listing / FDA GUDID / peer-reviewed spec).

| Family (study role) | Contacts | Contact length | Contact = body Ø | Pitch | Contact metal |
|---|---|---|---|---|---|
| **Behnke-Fried** `BF10R-SP21X-0C3` (R.AMY, R.HIPPO, R.OF) | 10 | **1.57 mm** | **1.28 mm** | **5 mm** | Pt/Ir 90/10 |
| **RD** Reduced-Diameter Spencer (R.STI, R.AT) | 10 | 2.29 mm | 1.10 mm | 5 mm | Platinum |
| **SD** Spencer Depth | 4–12 | 2.29 mm | 1.12 mm | 5 mm | Platinum |
| DIXI Microdeep (secondary) | 5–18 | 2.0 mm | 0.8 mm | 3.5 mm | Pt/Ir |
| PMT Depthalon (secondary) | 8–16 | 2.0 mm | 0.8 mm | 3.5 mm | Platinum |

> **`SP21X` caveat:** for Behnke-Fried hybrids `SP##X` is an Ad-Tech *build* code, **not**
> 21 mm spacing — BF macros are fixed at flush 5 mm. Trust `electrode_config.json` (5 mm).
> The 9× 40 µm Pt/Ir microwires are ~30× thinner than a mesh element and are **omitted**
> from the FEM (they do not perturb the mm-scale macro field).

The bundled catalog (`seeg_catalog.json`) carries these per part number; `SEEGLead(part_number
=...)` resolves geometry, and `first_contact_depth_mm` places contact-0 proximal to the tip
with clinical distal-first numbering.

---

## 5. Electrode composition — is there a conductive core?

**No.** Per Karimi 2025 §2.2.1, Lempka 2013, and the Ad-Tech catalogue construction: each
Pt/Ir contact is an individual ring bonded to its **own** insulated lead wire inside the
polyurethane body; the wires are mutually isolated, so the body between contacts is a solid
insulator. A shared conductive spine would short all contacts and is physically wrong. The
model is therefore: solid metal contact discs (equipotential) + continuous insulating body +
optional glial sheath, **no core**. At the 5–9 kHz carrier the electrode–electrolyte double
layer (Lempka's lumped interface term) is largely shorted, so the PEC/equipotential contact
is a good approximation and the interface term is second-order.

---

## 6. Decision variables — values and sources

Every tunable quantity in the model, its default, its sweep range, and where the value comes
from. Local PDFs are in [`references/`](references/); `[ext]` = cited without a local copy.

### 6.1 Electrode geometry — `BF10R-SP21X-0C3` (study electrode)

| Variable | Value | Unit | Source |
|---|---|---|---|
| `n_contacts` | 10 | — | [Ad-Tech 2015](references/AD-TECHCatalogue-2015.pdf) `:1762`; `electrode_config.json` |
| `contact_len_mm` | 1.57 | mm | [Ad-Tech 2015](references/AD-TECHCatalogue-2015.pdf) `:1694` (RE08R `1.57×1.28`) |
| `contact_dia_mm` = `body_dia_mm` | 1.28 | mm | [Ad-Tech 2015](references/AD-TECHCatalogue-2015.pdf) `:1694`; BF macros flush with shaft `:1742` |
| `spacing_mm` (pitch) | 5.0 | mm | `electrode_config.json`; Ad-Tech "flush 5 mm" `:1742` |
| `first_contact_depth_mm` | 2.0 | mm | clinical convention (contact-0 proximal to tip; distal-first numbering) |
| microwires (9 × 40 µm Pt/Ir) | omitted | — | [neuralynx note](references/neuralynx_macro_micro_electrodes.md); ~30× sub-element, no mm-field effect |

### 6.2 Material conductivities (element tags 13 / 14 / 15)

| Variable | Default | Sweep / range | Unit | Source |
|---|---|---|---|---|
| `contact_sigma` (tag 13) | **1×10⁶** | 1×10⁴ – 5.8×10⁷ | S/m | equipotential/PEC limit — [Datta 2011](references/Datta-2011_BrainStimulation.pdf) (5.8×10⁷ physical), [Lempka 2013](references/Lempka-2013_PLoSONE.pdf), [Karimi 2025](references/Karimi_2025_J._Neural_Eng._22_016039.pdf) (PEC). 1×10⁶ ≥6 orders > tissue → equipotential while keeping the solve conditioned (Missey-2026 uses PEC) |
| `shaft_sigma` (tag 14) | **1×10⁻⁵** | 1×10⁻⁴ – 1×10⁻¹² | S/m | polyurethane insulator (physical ~1×10⁻¹²); Missey "inter-contacts as insulators"; ≥4 orders < tissue blocks current without ill-conditioning |
| `sheath_sigma` (tag 15) | **0.05** | 0.05 – 0.30 (acute 1.0–1.7) | S/m | [Karimi 2025](references/Karimi_2025_J._Neural_Eng._22_016039.pdf) mouse scar @1 kHz (0.019/0.05/0.28 @ 20 Hz/1 kHz/300 kHz). Corroborating: Grill&Mortimer 1994 (0.15) `[ext]`; Butson 2006 (0.05–0.20) `[ext]`; Yousif 2008 (0.125) `[ext]` |

### 6.3 Glial sheath geometry

| Variable | Default | Sweep / range | Unit | Source |
|---|---|---|---|---|
| `sheath_thickness_um` | **150** | 100 – 500 | µm | [Missey 2026 SI Fig S7](references/Missey-2026.pdf) / [mmc1.docx](references/mmc1.docx) (150–300); DBS/FBR lit 250–500 `[ext]` |
| encapsulation extent | complete tube, **all tissue** | brain-only via `sheath_tags=(1,2,3)` | — | this study's requirement (no bare GM/WM); enhancement decays <400 µm ≈ sheath ([Missey 2026](references/Missey-2026.pdf)) |
| shell mesh thickness | max(1 voxel, `sheath/voxel`) | — | voxel | thin-shell equivalence: σ scaled ×(t_mesh/t_phys) to preserve sheet resistance t/σ |

### 6.4 Reference tissue conductivities (SimNIBS, unchanged)

| Tissue | σ (S/m) | Source |
|---|---|---|
| White matter (tag 1) | 0.126 | [Saturnino 2019](references/Saturnino-2019_ConductivityUncertainty_NeuroImage.pdf); `mesh_element_properties.py` |
| Grey matter (tag 2) | 0.275 | id. — sheath σ < GM σ ⇒ resistive shell (drives the near-contact enhancement) |
| CSF (tag 3) | 1.654 | id. |

### 6.5 Mesh resolution (electrode fidelity vs cost)

| Variable | Default (`CONF_FINE`) | Alt presets | Source / rationale |
|---|---|---|---|
| `label_voxel_mm` | 0.25 | 0.30 (`CONF_STANDARD`), 0.20 (`CONF_ULTRA`) | electrode surface fidelity, RMS radial err ≈ ½·voxel = 0.12 mm |
| `electrode_edge_mm` | 0.12 | — | fine tet size at the electrode |
| `electrode_facet_distance_mm` | 0.03 | — | electrode surface boundary tolerance |
| tissue `elem_sizes` | charm native (`standard[1,5]`, WM`[1,7]`, GM`[1,2]`, scalp`[1,10]`) | — | [charm.ini](../../charm.ini); [Puonti 2020](references/Puonti-2020_NeuroImage.pdf) — keeps the head native |

### 6.6 Solver & validation conventions

| Decision | Value | Source |
|---|---|---|
| Field equation | purely resistive QSA `∇·(σ∇φ)=0` (no permittivity) | [Opitz 2016](references/Opitz-2016_ScientificReports.pdf); [Missey 2026](references/Missey-2026.pdf); QSA valid 1–9 kHz (σ≫ωε) |
| Contact boundary | high-σ volume ≈ floating equipotential (net-≈0 current for recording) | [Datta 2011](references/Datta-2011_BrainStimulation.pdf); [Lempka 2013](references/Lempka-2013_PLoSONE.pdf) |
| Validation metrics | Pearson **r** (spatial distribution), regression slope **s** (magnitude) | [Huang 2017](references/Huang-2017_eLife.pdf) |
| TI principle / carriers | AM envelope of two kHz carriers | [Grossman 2017](references/Grossman-2017_Cell.pdf) |
| Passive-sample baseline | read φ at node nearest each contact, then symmetric difference quotient | [Huang 2017](references/Huang-2017_eLife.pdf); [Opitz 2016](references/Opitz-2016_ScientificReports.pdf); [Louviot 2022](references/Louviot-2022_BrainStimulation.pdf) |

---

## 7. Resolution & honest limits

- **Electrode fidelity is bounded by `label_voxel_mm`** (a voxel mesher can't beat its label
  staircase; RMS radial error ≈ ½·voxel). 0.25 mm → ~0.12 mm rod error.
- **The 150 µm sheath is sub-voxel** on any tractable whole-head grid, so it is meshed as a
  ≥1-voxel shell with σ scaled to preserve its sheet resistance (t/σ). This is a thin-shell
  *electrical* equivalent, not a literal 150 µm layer. A true-thickness sheath needs
  `preview_electrode` (standalone, 40 µm) or a surface-corefine route (staged).
- **Cost:** native `create_mesh` preprocessing is voxel-bound — whole head at 0.25 mm ≈
  90–120 min. Coarsen `label_voxel_mm` to trade electrode crispness for speed.

---

## 8. Deployment notes

- The fork is an unbuilt source tree. Verification is done against the installed **SimNIBS
  4.6** env (the seeg package symlinked into its site-packages; the three charm files patched
  with `.bak_preseeg` backups). Tags 13/14/15 self-register at runtime.
- The `charm --seeg` path writes a `*_seeg.json` sidecar (leads, materials, voxel counts,
  warnings) next to the mesh for provenance.
- To solve, feed `pl.mesh` + `pl.cond_list` (or the tagged `ernie.msh`) to the stock
  `run_simnibs`; the new tags are already in `standard_cond()`.

---

## 9. References

Local PDFs in [`references/`](references/):

- **Ad-Tech Medical, Product Catalogue 2015** — electrode geometry (contact length/diameter,
  spacing, contact counts). `references/AD-TECHCatalogue-2015.pdf`
- **Karimi F, et al. 2025**, *J Neural Eng* 22:016039 — implanted-electrode FEM model (PEC
  contacts, insulating shaft, scar σ). `references/Karimi_2025_J._Neural_Eng._22_016039.pdf`
- **Lempka SF, McIntyre CC 2013**, *PLoS ONE* 8:e59839 — depth-electrode encapsulation shell,
  contact interface, electrostatic≈electrodynamic. `references/Lempka-2013_PLoSONE.pdf`
- **Datta A, et al. 2011**, *Brain Stimul* 4:169 — modeled electrode metal σ=5.8×10⁷, Neumann
  insulation. `references/Datta-2011_BrainStimulation.pdf`
- **Missey F, et al. 2026**, *Brain Stimulation* — human hippocampal TI; SI Fig S7 glial
  sheath 150–300 µm, 2–7.2× enhancement decaying <400 µm; AM 230 ms method.
  `references/Missey-2026.pdf`, `references/mmc1.docx`
- **Huang Y, et al. 2017**, *eLife* 6:e18834 — intracranial tES field validation; r/s metrics;
  passive node sampling; symmetric difference quotient. `references/Huang-2017_eLife.pdf`
- **Opitz A, et al. 2016**, *Sci Rep* 6:31236 — depth-electrode tES fields; QSA. 
  `references/Opitz-2016_ScientificReports.pdf`
- **Louviot S, et al. 2022**, *Brain Stimul* 15:1 — deep-structure tES fields; DIXI geometry.
  `references/Louviot-2022_BrainStimulation.pdf`
- **Saturnino GB, et al. 2019**, *NeuroImage* — SimNIBS conductivity uncertainty (tissue σ).
  `references/Saturnino-2019_ConductivityUncertainty_NeuroImage.pdf`
- **Puonti O, et al. 2020**, *NeuroImage* — charm segmentation & meshing.
  `references/Puonti-2020_NeuroImage.pdf`
- **Grossman N, et al. 2017**, *Cell* 169:1029 — temporal interference principle.
  `references/Grossman-2017_Cell.pdf`

Cited without a local copy (secondary sheath-σ sources):

- Grill WM, Mortimer JT 1994, *Ann Biomed Eng* 22:23 — measured encapsulation σ (0.15 S/m),
  frequency-independence.
- Butson CR, Maks CB, McIntyre CC 2006, *Clin Neurophysiol* 117:447 — DBS encapsulation
  0–1 mm / σ 0.05–0.20 S/m.
- Yousif N, et al. 2008, *Brain Res Bull* 74:361 — peri-electrode space; acute 1.7 / chronic
  0.125 S/m.
