import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_02 = ROOT / "notebooks" / "02_rose_vs_lrom_cross_sections.ipynb"
BENCHMARK_03 = (
    ROOT / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_03.ipynb"
)

HEADINGS = (
    "# 02. ROSE and LROM Cross-Section Comparison",
    "## 1. Physical Problem and Optical-Potential Parameters",
    "## 2. Shared Training and Testing Samples",
    "## 3. Potential Variation and LROM Predictor Locations",
    "## 4. Equal-Basis ROSE and LROM Emulators",
    "## 5. Representative Cross-Section Predictions",
    "## 6. Cross-Section Error Distributions",
    "## 7. Basis and Operator-Size Comparison",
    "## 8. Accuracy Versus Online Time",
    "## 9. Validation Summary",
)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def notebook_text(path: Path) -> str:
    notebook = load_notebook(path)
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def code_sources(path: Path) -> list[str]:
    notebook = load_notebook(path)
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook02_shell_contract() -> None:
    assert NOTEBOOK_02.exists()
    text = notebook_text(NOTEBOOK_02)
    for heading in HEADINGS:
        assert heading in text
    assert "import lrom_legacy.v2_0 as lrom" in text
    assert "TARGET = (40, 20)" in text
    assert "PROJECTILE = (1, 0)" in text
    assert "LAB_ENERGY = 14.1" in text
    assert "L_MAX = 3" in text
    assert "MESH_SIZE = 600" in text
    assert "N_TRAIN = 200" in text
    assert "N_TEST = 100" in text
    assert "HALF_WIDTH = 0.20" in text
    assert "BASIS_SIZES = (4, 6, 8)" in text
    assert "ROSE_EIM_SIZES = (4, 8, 12)" in text
    assert "LROM_PREDICTOR_COUNTS = (4, 8, 12)" in text
    assert "SEED = 1204" in text


def test_notebook02_code_cells_compile() -> None:
    for index, source in enumerate(code_sources(NOTEBOOK_02)):
        compile(source, f"{NOTEBOOK_02.name} code cell {index}", "exec")
