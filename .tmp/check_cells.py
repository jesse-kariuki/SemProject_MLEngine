"""Execute selected non-pipeline cells of pipeline.ipynb (fast check of the new analysis cells)."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WANTED = {"setup", "sec2-eda", "sec4-evidence", "results-tables"}
nb = json.loads((ROOT / "pipeline.ipynb").read_text(encoding="utf-8"))

ns = {"__name__": "__main__"}
for c in nb["cells"]:
    if c["cell_type"] != "code" or c.get("id") not in WANTED:
        continue
    src = "".join(c["source"])
    print(f"\n===== CELL {c['id']} =====", flush=True)
    try:
        exec(compile(src, f"<cell {c['id']}>", "exec"), ns)
    except BaseException:
        traceback.print_exc()
        print(f"\nFAILED IN CELL {c['id']}", flush=True)
        sys.exit(1)

print("\nANALYSIS CELLS OK", flush=True)
