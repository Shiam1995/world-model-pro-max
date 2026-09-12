"""Train several policies from different seeds and validate each on the real ROM.

Different seeds give genuinely independent drivers: together they are the
ensemble whose disagreement is the confidence signal, and separately they tell
us whether one good lap was luck.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.obs.track import Track
from src.sim.kart import KartSim, Physics
from src.sim.train_es import evolve

SEEDS = [1, 2, 3, 4]

def main():
    d = json.load(open("tracks/luigi_raceway.json"))
    tr = Track.from_lap(np.array(d["points"], float), spacing=150., smooth_window=5)
    Path("policies").mkdir(exist_ok=True)
    out = {}
    for s in SEEDS:
        sim = KartSim(tr, Physics.calibrated(), n=384, seed=s)
        t0 = time.time()
        mean, best = evolve(sim, generations=80, frames=900, seed=s, verbose=False)
        laps = best[1] / tr.length
        np.save(f"policies/es_seed{s}.npy", mean)
        out[s] = {"sim_laps": round(float(laps), 3), "train_s": round(time.time()-t0)}
        print(f"seed {s}: {laps:.2f} laps in sim, {out[s]['train_s']}s", flush=True)
    Path("runs/fleet_train.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
