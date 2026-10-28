"""Execute every code cell of pipeline.ipynb in order, in one process (headless check of the notebook)."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "pipeline.ipynb").read_text(encoding="utf-8"))

start = None
if len(sys.argv) > 1 and sys.argv[1] == "--from":
    start = sys.argv[2]
start_idx = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == start), 0) if start else 0

ns = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    if i < start_idx and c.get("id") != "setup":  # always run setup for its helpers
        continue
    src = "".join(c["source"])
    print(f"\n===== CELL {c['id']} =====", flush=True)
    try:
        exec(compile(src, f"<cell {c['id']}>", "exec"), ns)
    except BaseException:
        traceback.print_exc()
        print(f"\nNOTEBOOK RUN FAILED IN CELL {c['id']}", flush=True)
        sys.exit(1)

print("\nNOTEBOOK RUN COMPLETE", flush=True)
