"""Does a policy drive a track it has never seen?

The whole project rests on this. The observation vector deliberately carries no
absolute position, no lap counter and no course identity, so in principle a
policy trained on one road can drive another. This measures whether that is
actually true, on fifteen Mario Kart courses the policy never trained on — which
is the same question as "will it drive the Unreal map".

Zero-shot is compared against a policy trained natively on each course, so the
gap is a number rather than an impression.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.obs.track import Track
from src.sim.kart import KartSim, Physics
from src.sim.train_es import act, evolve

FRAMES = 1200


def load_track(name):
    d = json.load(open(f"tracks/{name}.json"))
    return Track.from_lap(np.array(d["points"], float), spacing=150., smooth_window=5)


def evaluate(theta, track, starts=6, frames=FRAMES):
    """Average over several start points so one lucky launch cannot carry it."""
    laps, off = [], []
    n_cp = len(track.centre)
    for k in range(starts):
        sim = KartSim(track, Physics.calibrated(), n=1, seed=100 + k)
        obs = sim.reset(start_cp=int(k * n_cp / starts))
        tot = 0.0; o = 0
        for _ in range(frames):
            obs = sim.step(act(theta, obs, obs.shape[1]))
            tot += float(sim.progress()[0])
            o += int(abs(sim._project(update=False)[2][0]) > sim.p.half_width)
        laps.append(tot / track.length)
        off.append(o / frames)
    return float(np.mean(laps)), float(np.mean(off))


def main():
    names = sorted(p.stem for p in Path("tracks").glob("*.json"))
    home = np.load("policies/luigi_raceway_v2.npy")
    rows = {}
    print(f"{'course':22s} {'zero-shot':>10} {'off':>6} | {'native':>8} {'off':>6} | {'retained':>9}")
    for n in names:
        tr = load_track(n)
        zl, zo = evaluate(home, tr)
        sim = KartSim(tr, Physics.calibrated(), n=256, seed=7)
        t0 = time.time()
        native, _ = evolve(sim, generations=45, frames=900, seed=7, verbose=False)
        nl, no = evaluate(native, tr)
        np.save(f"policies/native_{n}.npy", native)
        ret = zl / nl if nl > 0.01 else 0.0
        rows[n] = {"zero_shot_laps": zl, "zero_shot_offroad": zo,
                   "native_laps": nl, "native_offroad": no, "retained": ret,
                   "train_s": round(time.time() - t0)}
        print(f"{n:22s} {zl:10.2f} {zo:6.1%} | {nl:8.2f} {no:6.1%} | {ret:8.0%}",
              flush=True)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/generalisation.json").write_text(json.dumps(rows, indent=1))
    unseen = {k: v for k, v in rows.items() if k != "luigi_raceway"}
    print(f"\nacross {len(unseen)} unseen courses:")
    print(f"  zero-shot laps  : {np.mean([v['zero_shot_laps'] for v in unseen.values()]):.2f}")
    print(f"  native laps     : {np.mean([v['native_laps'] for v in unseen.values()]):.2f}")
    print(f"  performance kept: {np.mean([v['retained'] for v in unseen.values()]):.0%}")
    print(f"  courses where zero-shot completes >=1 lap: "
          f"{sum(1 for v in unseen.values() if v['zero_shot_laps'] >= 1.0)}/{len(unseen)}")


if __name__ == "__main__":
    main()
