"""Screenshot selected executed notebook cells (code + real stdout/table output) as PNGs.

  python scripts/render_cells.py OUT_DIR

Uses headless Google Chrome. stderr streams and vLLM's log lines
(INFO/WARNING/engine logs, which vLLM prints to stdout) are omitted for readability;
everything else is the cell's saved output, unedited.
"""
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "screenshots")
OUT.mkdir(parents=True, exist_ok=True)

# output file -> code cells (by the "# [..]" / first-line marker) to put in one screenshot
SHOTS = {
    "env.png": ["[单元格 2]"],
    "step1-lab.png": ["from mini_scientist import Lab, RULES, RULES_BY_NAME"],
    "step2-sandbox.png": ["from mini_scientist.lab import compile_expr, ExprError"],
    "step3-dryrun.png": ["from mini_scientist import Episode, run_batch"],
    "step4-model.png": ["from mini_scientist.backend import VLLMBackend"],
    "step5-episode.png": ["def show(ep):"],
    "step6-batch.png": ["def evaluate(backend", "n_lines = sum("],
    "step7-failures.png": ["def own_data_conflicts"],
    "step8-gate.png": ["eps_g, df_g, sec_g"],
    "baseline.png": ["from mini_scientist.baseline import"],
    "exceptions.png": ["# 异常验收演示"],
}

CSS = """
body{margin:0;padding:18px;background:#fff;font:14px/1.5 'Noto Sans CJK SC','Droid Sans Fallback','Noto Emoji',sans-serif;color:#1f1f1f;width:1000px}
.cell{border:1px solid #e1e0d9;border-radius:6px;margin-bottom:14px;overflow:hidden}
.in{background:#f6f8fa;padding:10px 14px;border-bottom:1px solid #e1e0d9}
.out{padding:10px 14px}
.lbl{color:#898781;font:12px monospace;margin-bottom:4px}
pre{margin:0;white-space:pre-wrap;word-break:break-all;font:12.5px/1.45 'DejaVu Sans Mono','Droid Sans Fallback','Noto Emoji',monospace}
table{border-collapse:collapse;font:12.5px 'DejaVu Sans Mono','Droid Sans Fallback','Noto Emoji',monospace}
td,th{border:1px solid #e1e0d9;padding:3px 8px;text-align:right}
th{background:#f6f8fa}
"""


LOG = re.compile(r"^(INFO|WARNING|DEBUG|\x1b|\[Gloo\]|\(EngineCore)")


def strip_logs(text):
    """Drop vLLM / NCCL log lines that vLLM prints to stdout; keep the cell's own prints."""
    return "".join(l for l in text.splitlines(keepends=True) if not LOG.match(l))


def cell_html(c):
    src = "".join(c["source"])
    parts = [f'<div class="cell"><div class="in"><div class="lbl">In [{c.get("execution_count")}]</div>'
             f"<pre>{html.escape(src)}</pre></div>"]
    outs = []
    for o in c.get("outputs", []):
        if o["output_type"] == "stream" and o["name"] == "stdout":
            outs.append(f"<pre>{html.escape(strip_logs(''.join(o['text'])))}</pre>")
        elif o["output_type"] in ("execute_result", "display_data"):
            d = o["data"]
            outs.append("".join(d["text/html"]) if "text/html" in d else f"<pre>{html.escape(''.join(d.get('text/plain', '')))}</pre>")
        elif o["output_type"] == "error":
            outs.append(f"<pre>{html.escape(o['ename'] + ': ' + o['evalue'])}</pre>")
    if outs:
        parts.append(f'<div class="out"><div class="lbl">Out</div>{"".join(outs)}</div>')
    return "".join(parts) + "</div>"


def trim(path):
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im.crop((0, 0, min(im.width, box[2] + 18), box[3] + 18)).save(path)


nb = json.loads((ROOT / "mini_ai_scientist.ipynb").read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
for name, markers in SHOTS.items():
    cells = [next(c for c in code if m in "".join(c["source"])) for m in markers]
    doc = f"<!doctype html><meta charset=utf-8><style>{CSS}</style>" + "".join(cell_html(c) for c in cells)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(doc)
    lines = sum(doc.count("\n") for _ in [0]) + 40
    target = OUT / name
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    f"--window-size=1040,{min(8000, 22 * lines)}", f"--screenshot={target}", f"file://{f.name}"],
                   check=True, capture_output=True)
    trim(target)
    print(name, Image.open(target).size)
