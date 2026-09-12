"""Drive an Unreal vehicle over a socket. One JSON object per line, both ways.

    python3 scripts/unreal_server.py --policy policies/luigi_raceway_v2.npy --port 8791

Unreal connects, sends one {"type":"track", ...}, then one {"type":"state", ...}
per tick and applies the {"steer", "throttle", "brake"} that comes back.
"""
import argparse, json, socket, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.obs.unreal_adapter import UnrealAdapter
from src.policy.drive import Driver


def serve(policy, port, handedness):
    drv = Driver.load(policy)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(1)
    print(f"waiting for Unreal on 127.0.0.1:{port} (policy {drv.name})", flush=True)
    conn, _ = srv.accept()
    print("connected", flush=True)
    f = conn.makefile("rw")
    ad = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("type") == "track":
            ad = UnrealAdapter(msg["centreline"],
                               up_axis=msg.get("up_axis", "z"),
                               units_per_metre=msg.get("units_per_metre", 100.0),
                               half_width=msg.get("half_width"),
                               handedness=handedness)
            print(f"track: {ad.track.n_checkpoints} checkpoints, "
                  f"{ad.track.length:.0f} internal units", flush=True)
            f.write(json.dumps({"ok": True, "checkpoints": ad.track.n_checkpoints}) + "\n")
            f.flush()
            continue
        if ad is None:
            f.write(json.dumps({"error": "send a track message first"}) + "\n"); f.flush()
            continue
        obs = ad.observe(msg["pos"], msg["vel"])
        f.write(json.dumps({"steer": drv.steer(obs), "throttle": 1.0, "brake": 0.0}) + "\n")
        f.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policies/luigi_raceway_v2.npy")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--handedness", type=int, default=1, choices=(1, -1),
                    help="flip to -1 if +steer turns the car the wrong way")
    a = ap.parse_args()
    serve(a.policy, a.port, a.handedness)
