"""Execute mini_ai_scientist.ipynb top to bottom and save it with real outputs.

  python scripts/execute_notebook.py [--kernel python3]
"""
import argparse
import json
import time
from pathlib import Path

from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--kernel", default="python3")
args = ap.parse_args()

path = ROOT / "mini_ai_scientist.ipynb"
nb_json = json.loads(path.read_text())
import nbformat  # noqa: E402  (ships with nbclient)
nb = nbformat.reads(json.dumps(nb_json), as_version=4)

t0 = time.time()
client = NotebookClient(nb, timeout=7200, kernel_name=args.kernel, resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
    print(f"saved {path} after {time.time() - t0:.0f}s")
