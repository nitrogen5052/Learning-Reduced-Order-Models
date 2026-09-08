"""Execute all collaborator notebooks in order."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("LROM_BACKEND", "cpu")
for path in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
    print(f"Executing {path.name}", flush=True)
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    print(f"Finished {path.name}", flush=True)

