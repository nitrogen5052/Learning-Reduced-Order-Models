"""Read the vendored curated neutron-elastic JSON records."""

from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

_SYMBOLS = (
    "n","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U",
)
_Z = {symbol:z for z,symbol in enumerate(_SYMBOLS) if z}
_TARGET = re.compile(r"^([A-Z][a-z]?)-(\d+)$")
_ROLE = {"kduq":"fit", "test":"held_out", "chuq":"historical_validation", "elm":"supplementary_validation"}


def parse_target(target):
    match = _TARGET.match(str(target))
    if match is None or match.group(1) not in _Z or int(match.group(2)) <= 0:
        raise ValueError(f"not a definite isotope: {target!r}")
    return int(match.group(2)), _Z[match.group(1)]


def _vector(value, length):
    result = np.asarray(value if value is not None else np.full(length, np.nan), float)
    return np.full(length, float(result)) if result.ndim == 0 else result


def load_neutron_elastic(root, corpora=("kduq", "test")):
    """Return point-level and record-level tables with b/sr converted to mb/sr."""
    root = Path(root).resolve()
    point_rows, record_rows = [], []
    for corpus in corpora:
        directory = root / f"data/{corpus}/neutron_elastic"
        if not directory.is_dir():
            raise FileNotFoundError(f"missing curated directory: {directory}")
        for path in sorted(directory.glob("*.json")):
            for local_index, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
                if record.get("projectile") != "neutron" or record.get("type") != "ECS":
                    continue
                target = record.get("target", path.stem.replace("_", "-"))
                try:
                    a,z = parse_target(target); definite = True
                except ValueError:
                    a,z,definite = 0,0,False
                if (record.get("energy_units"),record.get("x_units"),record.get("y_units")) != ("MeV","CM-degrees","b/sr"):
                    raise ValueError(f"unexpected units in {path}")
                data = record["data"]
                angles = np.asarray(data["x"],float)
                values, errors = _vector(data["y"],len(angles)), _vector(data.get("y_err"),len(angles))
                energy = float(record["energy"])
                accession = str(record.get("EXFORAccessionNumber", ""))
                record_id = f"{corpus}:{accession}:{target}:{energy:.9g}:{local_index}"
                norm = float(data.get("systematic_normalization_error", .05))
                quasi = record.get("summed_excitation_max_MeV") is not None
                base = dict(record_id=record_id,corpus=corpus,role=_ROLE[corpus],target=target,A=a,Z=z,definite_isotope=definite,energy_MeV=energy,normalization_fraction=norm,quasi_elastic=quasi,EXFOR_accession=accession)
                record_rows.append(base | {"points":len(angles),"source_file":str(path.relative_to(root))})
                for i,(angle,value,error) in enumerate(zip(angles,values,errors)):
                    point_rows.append(base | dict(point_index=i,angle_cm_deg=float(angle),cross_section_mb_sr=1000*float(value),point_error_mb_sr=1000*float(error)))
    return pd.DataFrame(point_rows), pd.DataFrame(record_rows)


def strict_lrom_selection(points, energy=(5.0,200.0)):
    lo,hi = energy
    selected = points[
        points.definite_isotope & ~points.quasi_elastic
        & points.energy_MeV.between(lo,hi,inclusive="both")
        & points.angle_cm_deg.between(1,179,inclusive="both")
        & np.isfinite(points.cross_section_mb_sr) & (points.cross_section_mb_sr>0)
        & np.isfinite(points.point_error_mb_sr) & (points.point_error_mb_sr>0)
    ].copy()
    return selected.sort_values(["corpus","A","Z","energy_MeV","record_id","angle_cm_deg"]).reset_index(drop=True)

