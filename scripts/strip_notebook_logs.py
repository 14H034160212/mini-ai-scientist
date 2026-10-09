"""Remove vLLM/NCCL log lines and stderr streams from a saved notebook's outputs.

These logs carry machine-specific paths and addresses; the cells' own prints,
tables and results are kept unchanged.  python scripts/strip_notebook_logs.py
"""
import json
import re
from pathlib import Path

LOG = re.compile(r"^(INFO|WARNING|DEBUG|\x1b|\[Gloo\]|\(EngineCore|\[rank\d)")
path = Path(__file__).resolve().parents[1] / "mini_ai_scientist.ipynb"
nb = json.loads(path.read_text())
removed = 0
for cell in nb["cells"]:
    outs = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            if o["name"] == "stderr":
                removed += 1
                continue
            text = "".join(o["text"]) if isinstance(o["text"], list) else o["text"]
            kept = [l for l in text.splitlines(keepends=True) if not LOG.match(l)]
            removed += text.count("\n") - sum(l.count("\n") for l in kept)
            if not kept:
                continue
            o["text"] = kept
        outs.append(o)
    if cell["cell_type"] == "code":
        cell["outputs"] = outs
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
print(f"removed {removed} log lines/streams")
