"""Electrode catalog: resolve a part number to geometry + material properties.

The catalog is a flat ``{part_number: ElectrodeSpec-fields}`` mapping shipped as
``seeg_catalog.json`` inside the package (loaded via ``importlib.resources`` so it
works from an installed / editable dependency). Users can point at their own JSON.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields
from importlib import resources
from typing import Any

_CATALOG_RESOURCE = "seeg_catalog.json"


@dataclass(frozen=True)
class ElectrodeSpec:
    """Geometry + material properties of one sEEG electrode type."""

    part_number: str
    manufacturer: str
    family: str
    n_contacts: int
    contact_len_mm: float
    contact_dia_mm: float
    shaft_dia_mm: float
    spacing_mm: float
    contact_material: str = ""
    shaft_material: str = ""
    contact_sigma_S_per_m: float = 9.43e6
    shaft_sigma_S_per_m: float = 1e-12
    body_dia_mm: float | None = None       # constant outer Ø of the one-piece body; None -> contact_dia
    microwires: dict | None = None
    notes: str = ""

    @property
    def body_radius_mm(self) -> float:
        # A real sEEG electrode is a CONSTANT-diameter one-piece insulating body with flush
        # metal rings; contacts differ only axially, never radially (Karimi 2025). So the
        # outer rod radius is a single value -- use body_dia if given, else the contact Ø.
        dia = self.body_dia_mm if self.body_dia_mm is not None else self.contact_dia_mm
        return 0.5 * dia

    @property
    def contact_radius_mm(self) -> float:
        return 0.5 * self.contact_dia_mm

    @property
    def shaft_radius_mm(self) -> float:
        return 0.5 * self.shaft_dia_mm

    def __post_init__(self) -> None:
        for name, val in (
            ("n_contacts", self.n_contacts),
            ("contact_len_mm", self.contact_len_mm),
            ("contact_dia_mm", self.contact_dia_mm),
            ("shaft_dia_mm", self.shaft_dia_mm),
            ("spacing_mm", self.spacing_mm),
        ):
            if val is None or float(val) <= 0:
                raise ValueError(
                    f"ElectrodeSpec[{self.part_number!r}]: {name} must be positive, got {val!r}"
                )


_SPEC_FIELDS = {f.name for f in fields(ElectrodeSpec)} - {"part_number"}
_REQUIRED_FIELDS = {
    f.name
    for f in fields(ElectrodeSpec)
    if f.name != "part_number"
    and f.default is MISSING
    and f.default_factory is MISSING
}


def _spec_from_dict(part_number: str, d: dict[str, Any]) -> ElectrodeSpec:
    missing = sorted(_REQUIRED_FIELDS - d.keys())
    if missing:
        raise ValueError(
            f"catalog entry {part_number!r} is missing required field(s): "
            f"{missing}"
        )
    kwargs = {k: v for k, v in d.items() if k in _SPEC_FIELDS}
    return ElectrodeSpec(part_number=part_number, **kwargs)


class ElectrodeCatalog:
    """Lookup table keyed by manufacturer part number."""

    def __init__(self, path: str | None = None):
        if path is None:
            with resources.files("simnibs.simulation.seeg").joinpath(
                _CATALOG_RESOURCE
            ).open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        self._specs: dict[str, ElectrodeSpec] = {
            pn: _spec_from_dict(pn, d)
            for pn, d in raw.items()
            if not pn.startswith("_")
        }

    @classmethod
    def default(cls) -> "ElectrodeCatalog":
        return cls(path=None)

    def __getitem__(self, part_number: str) -> ElectrodeSpec:
        try:
            return self._specs[part_number]
        except KeyError:
            raise KeyError(
                f"part number {part_number!r} not in catalog. Known parts: {self.parts()}"
            ) from None

    def get(self, part_number: str, default=None):
        return self._specs.get(part_number, default)

    def parts(self) -> list[str]:
        return sorted(self._specs)

    def __contains__(self, part_number: str) -> bool:
        return part_number in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"ElectrodeCatalog({len(self)} parts: {self.parts()})"
