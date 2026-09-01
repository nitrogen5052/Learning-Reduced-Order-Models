"""Reader for the curated nucleon--nucleus JSON corpus.

The upstream repository has already performed the EXFOR curation.  We treat
its JSON files as immutable input and preserve source metadata while converting
the published differential cross sections from b/sr to mb/sr.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_SYMBOLS = (
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U",
)
_Z_BY_SYMBOL = {symbol: z for z, symbol in enumerate(_SYMBOLS) if z}
_TARGET = re.compile(r"^([A-Z][a-z]?)-(\d+)$")
_ROLE = {
    "kduq": "fit", "test": "held_out",
    "chuq": "historical_validation", "elm": "supplementary_validation",
}


def parse_target(target: str) -> tuple[int, int]:
    """Return ``(A, Z)`` for a definite isotope such as ``Pb-208``."""
    match = _TARGET.match(str(target))
    if match is None or match.group(1) not in _Z_BY_SYMBOL:
        raise ValueError(f"not a definite isotope target: {target!r}")
    return int(match.group(2)), _Z_BY_SYMBOL[match.group(1)]


def load_neutron_elastic(root, corpora=("kduq", "test")):
    """Return point-level and record-level neutron elastic tables."""
    root = Path(root).expanduser().resolve()
    if not (root / "data").is_dir():
        raise FileNotFoundError(
            f"curated corpus not found at {root}; set SCATTERING_DATA_ROOT"
        )
    point_rows, record_rows = [], []
    for corpus in corpora:
        directory = root / (
            "data/elm/elastic_diff_xs" if corpus == "elm"
            else f"data/{corpus}/neutron_elastic"
        )
        for path in sorted(directory.glob("*.json")):
            for local_index, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
                if record.get("projectile") != "neutron" or record.get("type") != "ECS":
                    continue
                target = record.get("target", path.stem.replace("_", "-"))
                try:
                    a, z = parse_target(target); definite = True
                except ValueError:
                    a, z, definite = 0, 0, False
                if (record.get("energy_units"), record.get("x_units"), record.get("y_units")) != ("MeV", "CM-degrees", "b/sr"):
                    raise ValueError(f"unexpected units in {path}")
                data = record["data"]
                angles = np.asarray(data["x"], float)
                values = np.asarray(data["y"], float)
                errors = np.asarray(data.get("y_err", np.full(len(angles), np.nan)), float)
                if errors.ndim == 0:
                    errors = np.full(len(angles), float(errors))
                energy = float(record["energy"])
                accession = str(record.get("EXFORAccessionNumber", ""))
                record_id = f"{corpus}:{accession}:{target}:{energy:.9g}:{local_index}"
                norm = float(data.get("systematic_normalization_error", 0.05))
                quasi = record.get("summed_excitation_max_MeV") is not None
                record_rows.append({
                    "record_id": record_id, "corpus": corpus, "role": _ROLE[corpus],
                    "target": target, "A": a, "Z": z, "definite_isotope": definite,
                    "energy_MeV": energy, "points": len(angles),
                    "normalization_fraction": norm, "quasi_elastic": quasi,
                    "EXFOR_accession": accession,
                    "source_file": str(path.relative_to(root)),
                })
                for i, (angle, value, error) in enumerate(zip(angles, values, errors)):
                    point_rows.append({
                        "record_id": record_id, "point_index": i,
                        "corpus": corpus, "role": _ROLE[corpus], "target": target,
                        "A": a, "Z": z, "definite_isotope": definite,
                        "energy_MeV": energy, "angle_cm_deg": float(angle),
                        "cross_section_mb_sr": 1000.0 * float(value),
                        "point_error_mb_sr": 1000.0 * float(error),
                        "normalization_fraction": norm, "quasi_elastic": quasi,
                        "EXFOR_accession": accession,
                    })
    return pd.DataFrame(point_rows), pd.DataFrame(record_rows)


def strict_lrom_selection(points, energy=(5.0, 200.0)):
    """Select definite-isotope, strictly elastic records in the LROM domain."""
    lo, hi = energy
    selected = points[
        points.definite_isotope & ~points.quasi_elastic
        & points.energy_MeV.between(lo, hi, inclusive="both")
        & points.angle_cm_deg.between(1.0, 179.0, inclusive="both")
        & np.isfinite(points.cross_section_mb_sr) & (points.cross_section_mb_sr > 0)
        & np.isfinite(points.point_error_mb_sr) & (points.point_error_mb_sr > 0)
    ].copy()
    return selected.sort_values(
        ["corpus", "A", "Z", "energy_MeV", "record_id", "angle_cm_deg"]
    ).reset_index(drop=True)

