"""CLI for the sEEG module (conforming build).

    simnibs_python -m simnibs.simulation.seeg conforming \
        --m2m m2m_ernie --leads leads.json --sheath-um 150 --preset fine \
        --out ernie_seeg_conforming.msh

    simnibs_python -m simnibs.simulation.seeg fields \
        --result-mesh TI.msh --leads leads.json --out axis_profiles.csv

``leads.json`` is a list of records:
    [{"name": "R.HIPPO", "part_number": "BF10R-SP21X-0C3",
      "entry_mm": [81.52, 4.10, 5.40], "target_mm": [29.86, 7.96, -1.42],
      "first_contact_depth_mm": 2.0, "n_contacts": 10}]
"""

from __future__ import annotations

import argparse
import json
import sys

from .geometry import SEEGLead
from .catalog import ElectrodeCatalog
from ._params import SEEGMaterials


def _load_leads(path: str) -> list[SEEGLead]:
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    if isinstance(records, dict):
        records = [records]
    return [
        SEEGLead(
            name=r["name"],
            part_number=r["part_number"],
            entry_mm=tuple(r["entry_mm"]),
            target_mm=tuple(r["target_mm"]),
            first_contact_depth_mm=float(r.get("first_contact_depth_mm", 2.0)),
            n_contacts=r.get("n_contacts"),
        )
        for r in records
    ]


def _cmd_conforming(args: argparse.Namespace) -> int:
    from dataclasses import replace
    from .embed import build_conforming_head
    from ._params import CONF_STANDARD, CONF_FINE, CONF_ULTRA

    leads = _load_leads(args.leads)
    catalog = ElectrodeCatalog(args.catalog) if args.catalog else None
    materials = SEEGMaterials(
        contact_sigma=args.contact_sigma,
        shaft_sigma=args.shaft_sigma,
        sheath_sigma=args.sheath_sigma,
        fibrous_sheath_sigma=args.fibrous_sheath_sigma,
    )
    res = {"standard": CONF_STANDARD, "fine": CONF_FINE, "ultra": CONF_ULTRA}[args.preset]
    over = {}
    if args.label_voxel_mm is not None:
        over["label_voxel_mm"] = args.label_voxel_mm
    if args.electrode_edge_mm is not None:
        over["electrode_edge_mm"] = args.electrode_edge_mm
    if over:
        res = replace(res, **over)

    print(f"[seeg] conforming build from {args.m2m} ({res}) -- heavy, minutes ...")
    pl = build_conforming_head(
        args.m2m, leads, sheath_thickness_um=args.sheath_um,
        materials=materials, catalog=catalog, resolution=res,
        mesher=args.mesher, num_threads=args.num_threads,
    )
    print(pl.summary())
    pl.mesh.write(args.out)
    print(f"[seeg] wrote {args.out}")
    for w in pl.warnings:
        print(f"[seeg]  ! {w}")
    return 0


def _cmd_fields(args: argparse.Namespace) -> int:
    import csv
    from simnibs.mesh_tools import mesh_io
    from .embed import SEEGPlacement
    from .sample import extract_axis_fields

    leads = _load_leads(args.leads)
    catalog = ElectrodeCatalog(args.catalog) if args.catalog else ElectrodeCatalog.default()
    centers = {ld.name: ld.contact_centers(catalog[ld.part_number]) for ld in leads}

    print(f"[seeg] reading result mesh {args.result_mesh}")
    rmesh = mesh_io.read_msh(args.result_mesh)
    placement = SEEGPlacement(
        mesh=rmesh, leads=leads, materials=SEEGMaterials(),
        sheath_thickness_mm=0.0, contact_centers=centers,
    )
    profiles = extract_axis_fields(
        rmesh, placement, field_name=args.field_name,
        grid_step_mm=args.grid_step, radial_max_mm=args.radial_max, catalog=catalog,
    )
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lead", "contact", "radial_mm", "value", "background", "enhancement"])
        for name, prof in profiles.items():
            enh = prof.enhancement()
            for ci in range(prof.radial_values.shape[0]):
                for ri, rmm in enumerate(prof.radial_mm):
                    w.writerow([name, ci, f"{rmm:.4f}", f"{prof.radial_values[ci, ri]:.6g}",
                                f"{prof.background:.6g}", f"{enh[ci]:.3f}"])
    for name, prof in profiles.items():
        enh = prof.enhancement()
        print(f"[seeg] {name}: peak enhancement {enh.min():.1f}-{enh.max():.1f}x, "
              f"background {prof.background:.4g}")
    print(f"[seeg] wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simnibs.simulation.seeg", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("conforming",
                        help="rebuild the whole head with conforming (smooth) sEEG electrodes")
    pc.add_argument("--m2m", required=True, help="m2m folder or a tissue-label .nii.gz")
    pc.add_argument("--leads", required=True, help="leads.json")
    pc.add_argument("--sheath-um", type=float, default=150.0)
    pc.add_argument("--preset", choices=["standard", "fine", "ultra"], default="fine")
    pc.add_argument("--label-voxel-mm", type=float, default=None,
                    help="whole-head label voxel = electrode geometric fidelity")
    pc.add_argument("--electrode-edge-mm", type=float, default=None,
                    help="target tet/facet size at the electrode")
    pc.add_argument("--mesher", choices=["create_mesh", "image2mesh"], default="create_mesh",
                    help="create_mesh (default, native-quality head, solver-ready) | "
                         "image2mesh (faster/low-RAM raw CGAL, non-native head)")
    pc.add_argument("--contact-sigma", type=float, default=SEEGMaterials().contact_sigma)
    pc.add_argument("--shaft-sigma", type=float, default=SEEGMaterials().shaft_sigma)
    pc.add_argument("--sheath-sigma", type=float, default=SEEGMaterials().sheath_sigma,
                    help="glial sheath sigma (brain GM/WM), S/m")
    pc.add_argument("--fibrous-sheath-sigma", type=float,
                    default=SEEGMaterials().fibrous_sheath_sigma,
                    help="fibrous sheath sigma (bone/scalp/soft-tissue tract), S/m")
    pc.add_argument("--catalog", default=None)
    pc.add_argument("--num-threads", type=int, default=8)
    pc.add_argument("--out", required=True)
    pc.set_defaults(func=_cmd_conforming)

    pf = sub.add_parser("fields", help="sample a solved field along/around lead(s)")
    pf.add_argument("--result-mesh", required=True, help="solved TI/TDCS mesh")
    pf.add_argument("--leads", required=True)
    pf.add_argument("--field-name", default=None)
    pf.add_argument("--grid-step", type=float, default=0.05)
    pf.add_argument("--radial-max", type=float, default=1.0)
    pf.add_argument("--catalog", default=None)
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=_cmd_fields)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
