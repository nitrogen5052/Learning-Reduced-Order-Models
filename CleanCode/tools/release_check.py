"""Fail if the collaborator release contains hidden workspace dependencies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("CodeRestructure", "ScatteringData", "\\Scratch\\", "/Scratch/")


def main() -> None:
    required = [
        ROOT / "notebooks/01_single_wavefunction.ipynb",
        ROOT / "notebooks/02_elastic_cross_sections.ipynb",
        ROOT / "notebooks/03_curated_data_and_global_lrom.ipynb",
        ROOT / "data/curated/data/kduq/neutron_elastic",
        ROOT / "data/curated/data/test/neutron_elastic",
        ROOT / "models/three_window/energy_window_0_b14_p20_ridge1e-14.pkl",
        ROOT / "models/three_window/energy_window_1_b14_p20_ridge1e-14.pkl",
        ROOT / "models/three_window/energy_window_2_b14_p20_ridge1e-14.pkl",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing release files: " + ", ".join(missing))

    scanned = []
    for suffix in ("*.py", "*.md", "*.toml", "*.json", "*.ipynb"):
        scanned.extend(ROOT.rglob(suffix))
    violations = []
    for path in scanned:
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in FORBIDDEN:
            # The README explicitly states that these paths are not required.
            if marker in text and path.name not in {"README.md", "release_check.py"}:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    if violations:
        raise SystemExit("hidden workspace references:\n" + "\n".join(violations))

    notebook_report = []
    for path in required[:3]:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        errors = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if errors:
            raise SystemExit(f"stored execution errors in {path.name}")
        images = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
        )
        notebook_report.append((path.name, len(notebook["cells"]), images))

    print(f"release root: {ROOT}")
    for name, cells, images in notebook_report:
        print(f"{name}: {cells} cells, {images} stored figures, no errors")
    print("curated data and all three pretrained windows are present")
    print("no hidden research-workspace dependency was found")


if __name__ == "__main__":
    main()
