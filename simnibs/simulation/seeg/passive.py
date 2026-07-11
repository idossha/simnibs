"""Passive-sample baseline (Huang-2017 method): read the field at the contact
locations from a solve that contains NO metal, to cross-check that the metallic
electrode's perturbation is negligible beyond the glial sheath.

The recorded quantity in a field-validation study is the potential difference
between neighbouring contacts; its spatial derivative along the electrode axis is
the axial E-field (the symmetric difference quotient used by seegkit / Huang 2017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from simnibs.mesh_tools import mesh_io
    from .embed import SEEGPlacement
    from .catalog import ElectrodeCatalog

_POTENTIAL_FIELDS = ("v", "V", "potential", "phi")


@dataclass
class PassiveProfile:
    lead_name: str
    contact_centers: np.ndarray     # (n, 3)
    potential: np.ndarray           # (n,) sampled potential at contacts
    axial_efield: np.ndarray        # (n,) symmetric-difference-quotient |dV/ds|


def _pick_potential(result_mesh: "mesh_io.Msh", field_name: str | None):
    fields = result_mesh.field
    if field_name is not None:
        return fields[field_name]
    for name in _POTENTIAL_FIELDS:
        if name in fields:
            return fields[name]
    raise KeyError(
        f"no potential field found; available: {list(fields.keys())}. "
        f"Pass field_name (the nodal potential 'v')."
    )


def passive_sample(
    result_mesh: "mesh_io.Msh",
    placement: "SEEGPlacement",
    field_name: str | None = None,
    catalog: "ElectrodeCatalog | None" = None,
) -> dict[str, "PassiveProfile"]:
    """Sample nodal potential at each contact centre and take the symmetric
    difference quotient along the axis to get the axial E-field (Huang 2017).

    ``result_mesh`` should be a metal-free solve (``materials=MATERIALS_PASSIVE`` or
    a plain head mesh) so this is the unperturbed reference.
    """
    fobj = _pick_potential(result_mesh, field_name)
    from .catalog import ElectrodeCatalog

    if catalog is None:
        catalog = ElectrodeCatalog.default()
    out: dict[str, PassiveProfile] = {}

    for lead in placement.leads:
        spec = catalog[lead.part_number]
        centers = placement.contact_centers[lead.name]
        v = np.asarray(
            fobj.interpolate_scattered(centers, out_fill=np.nan, squeeze=True)
        ).reshape(-1)

        # symmetric difference quotient along the axis (spacing = contact pitch)
        n = centers.shape[0]
        e = np.full(n, np.nan)
        h = spec.spacing_mm
        if n >= 3:
            e[1:-1] = np.abs(v[2:] - v[:-2]) / (2.0 * h)
        if n >= 2:
            e[0] = np.abs(v[1] - v[0]) / h
            e[-1] = np.abs(v[-1] - v[-2]) / h
        out[lead.name] = PassiveProfile(
            lead_name=lead.name,
            contact_centers=centers,
            potential=v,
            axial_efield=e,
        )
    return out
