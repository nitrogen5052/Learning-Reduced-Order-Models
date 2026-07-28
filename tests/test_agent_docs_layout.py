from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agent_documents_live_outside_git_checkout() -> None:
    assert not (ROOT / ".agents").exists()
    assert not (ROOT / "docs" / "superpowers").exists()
