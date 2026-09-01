"""Fetch the curated nucleon--nucleus corpus used by notebook 03."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


REPOSITORY = "https://github.com/beykyle/nucleon-nucleus-data.git"
DEFAULT_REVISION = "adc8558fe9fcf629af41bc909514f8da63d4f590"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "external" / "nucleon-nucleus-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION,
        help="checkout directory (default: external/nucleon-nucleus-data)",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help="upstream commit to fetch (default: the validated corpus revision)",
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()

    if (destination / "data").is_dir():
        print(f"Curated data already available at {destination}")
        if (destination / ".git").exists():
            commit = subprocess.check_output(
                ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
            ).strip()
            print(f"Upstream commit: {commit}")
        return
    if destination.exists():
        raise RuntimeError(
            f"{destination} exists but does not contain data/; move it or choose --destination"
        )
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch the curated data repository")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False)
    subprocess.run(["git", "-C", str(destination), "init"], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", REPOSITORY],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", args.revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
    if not (destination / "data").is_dir():
        raise RuntimeError("download completed, but the expected data/ directory is missing")
    commit = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"Curated data ready at {destination}")
    print(f"Upstream commit: {commit}")


if __name__ == "__main__":
    main()
