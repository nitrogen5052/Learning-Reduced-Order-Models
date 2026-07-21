from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".agents" / "validation" / "notebook01_rose_reference_diagnostic.py"


def test_diagnostic_bootstraps_the_repository_root_before_package_import() -> None:
    text = SCRIPT.read_text()

    assert "ROOT = Path(__file__).resolve().parents[2]" in text
    assert "sys.path.insert(0, str(ROOT))" in text
    assert text.index("sys.path.insert(0, str(ROOT))") < text.index(
        "import lrom"
    )


def test_diagnostic_owns_its_rose_eim() -> None:
    text = SCRIPT.read_text()

    assert "rose.InteractionEIMSpace(" in text
    assert "emulator.full_order_model[0].interaction" not in text
    assert "eim_basis_size" not in text
