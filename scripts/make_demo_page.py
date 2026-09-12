"""Build the demo page: the track, the driven line, and the telemetry."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.obs.track import Track

def build():
    d = json.load(open("tracks/luigi_raceway.json"))
    pts = np.array(d["points"], float)
    tr = Track.from_lap(pts, spacing=150., smooth_window=5)
    out = {
        "centre": [[float(p[0]), float(p[2])] for p in tr.centre],
        "length": float(tr.length),
        "checkpoints": int(tr.n_checkpoints),
        "half_width": 72.0,
        "runs": {},
    }
    for name, path in (("search", "runs/expert_lap.json"),
                       ("policy", "runs/transfer_eval.json")):
        f = Path(path)
        if not f.exists():
            continue
        blob = json.loads(f.read_text())
        log = blob.get("log", [])
        if not log:
            continue
        out["runs"][name] = {
            "xz": [[float(e["pos"][0]), float(e["pos"][2])] for e in log if "pos" in e],
            "speed": [float(e.get("speed", 0)) for e in log if "pos" in e],
            "summary": blob.get("summary", {k: blob[k] for k in
                                ("finished", "progress", "segments") if k in blob}),
        }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/demo_data.json").write_text(json.dumps(out))
    print(f"demo data: track {out['checkpoints']} cps, runs: {list(out['runs'])}")
    return out

if __name__ == "__main__":
    build()
