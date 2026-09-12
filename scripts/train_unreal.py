"""Train a policy native to an Unreal track, in the sim, against a fitted vehicle.

The third step of the loop the HANDOFF describes: record one open-loop calibration run in
Unreal, fit the sim to it, train here, drive the result back in Unreal.

    python3 scripts/train_unreal.py \
        --track ~/kx/unreal_track_kingscross_figure8_mk64.json \
        --physics runs/unreal_kx_physics.json \
        --out policies/kingscross_mk64_v1.npy

Training never touches Unreal. The sim runs ~500k kart-frames a second and the population
*is* the batch, so a generation is one vectorised rollout; Unreal manages about ten steps
a second. The whole reason for fitting a sim is that this step costs two minutes instead
of a week.

The centreline comes from the same `unreal_track_*.json` the level drives, converted with
the same length scale the adapter uses, so the track the policy learns is the track it
will be asked to drive -- including the flyover, whose elevation feeds the `grade` term.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.obs.track import Track                              # noqa: E402
from src.obs.unreal_adapter import units_per_metre_for_lap    # noqa: E402
from src.sim.kart import KartSim, Physics                     # noqa: E402
from src.sim.train_es import evolve, rollout, act, n_params    # noqa: E402


def load_track(path, units_per_metre, spacing_m=4.5, flip_y=True):
    """unreal_track_*.json -> a Track in MK64 units, in the adapter's frame.

    The adapter maps Unreal (x, y, z) to (x, elevation, z) with the ground plane in x/z,
    so the same mapping has to happen here or the policy trains on one axis convention
    and drives on another. flip_y matches the -trackflipy the level runs with: Unreal's
    OBJ importer negated Y, and the policy should see the world the car drives in.
    """
    doc = json.loads(Path(path).expanduser().read_text())
    sign = -1.0 if flip_y else 1.0
    points = []
    for sample in doc["centreline"]:
        x = float(sample["x"]) * units_per_metre
        y = float(sample["y"]) * sign * units_per_metre
        z = float(sample["z"]) * units_per_metre
        points.append([x, z, y])          # (x, elevation, z) — the adapter's ground frame
    spacing = spacing_m * units_per_metre
    return doc, Track.from_lap(np.array(points, dtype=float), spacing=spacing,
                               smooth_window=5)


def physics_from_fit(path, half_width_units):
    fit = json.loads(Path(path).expanduser().read_text())
    return Physics.calibrated(
        top_speed=float(fit["top_speed"]),
        accel_table=tuple(map(tuple, fit["accel_table"])),
        turn_table=tuple(map(tuple, fit["turn_table"])),
        steer_response=tuple(map(tuple, fit["steer_response"])),
        half_width=half_width_units,
        # No off-road speed penalty, because Unreal has none. Nothing in the level
        # changes friction at the kerb; what actually punishes leaving the carriageway
        # there is the 15 cm pavement step, 66 lamp columns, 49 trees and 102 parked
        # cars -- obstacles this sim cannot represent at all. Inventing a 0.35x
        # slowdown instead would train against a penalty that does not exist.
        #
        # Staying on the road is still trained for: train_es charges the fitness
        # 0.35 * offroad_frames * top_speed, which is a stated preference rather than
        # a claim about physics.
        off_road=1.0,
    ), fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=384)
    parser.add_argument("--frames", type=int, default=1400)
    parser.add_argument("--seeds", type=int, default=3,
                        help="independent runs; the best is saved, all are kept as an "
                             "ensemble because their disagreement is the confidence signal")
    parser.add_argument("--road-width-m", type=float, default=12.0)
    args = parser.parse_args()

    doc = json.loads(Path(args.track).expanduser().read_text())
    lap_m = float(doc["length_m"])
    units_per_metre = units_per_metre_for_lap(lap_m)
    half_width_units = (args.road_width_m / 2.0) * units_per_metre

    doc, track = load_track(args.track, units_per_metre)
    physics, fit = physics_from_fit(args.physics, half_width_units)

    print(f"track    {doc['name']}: lap {lap_m:.1f} m -> {track.length:.0f} MK64 units, "
          f"{len(track.centre)} checkpoints, spacing {track.spacing:.1f}")
    print(f"scale    {units_per_metre:.3f} units/m, half width {half_width_units:.1f} "
          f"units (MK64 Luigi is 72)")
    print(f"vehicle  top speed {physics.top_speed:.3f} units/tick, "
          f"{len(physics.accel_table)} accel bins, {len(physics.turn_table)} turn bins")

    results = []
    for seed in range(args.seeds):
        sim = KartSim(track, physics=physics, n=args.population, seed=seed)
        started = time.time()
        print(f"\n-- seed {seed} --", flush=True)
        mean, best = evolve(sim, generations=args.generations, frames=args.frames,
                            seed=seed, verbose=True)
        elapsed = time.time() - started

        # Score the trained mean on its own, from the start line, no jitter: that is the
        # number to compare against a lap, not the best of a jittered population.
        sim_eval = KartSim(track, physics=physics, n=1, seed=1234)
        fitness, progress, offroad = rollout(
            sim_eval, mean[None, :], frames=args.frames,
            jitter_lateral=0.0, jitter_yaw=0.0, start_cp=0)
        laps = float(progress[0]) / track.length
        print(f"seed {seed}: trained in {elapsed:.0f}s — clean run {laps:.2f} laps "
              f"({float(progress[0]):.0f} units), {float(offroad[0]):.0f} frames off road")
        results.append({"seed": seed, "theta": mean, "laps": laps,
                        "progress": float(progress[0]), "offroad": float(offroad[0]),
                        "train_seconds": elapsed})

    results.sort(key=lambda r: -r["laps"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, results[0]["theta"])
    for record in results[1:]:
        np.save(out.with_name(f"{out.stem}_seed{record['seed']}.npy"), record["theta"])

    meta = {
        "track": doc["name"],
        "track_file": str(Path(args.track).expanduser()),
        "lap_m": lap_m,
        "units_per_metre": units_per_metre,
        "half_width_units": half_width_units,
        "physics_file": str(Path(args.physics).expanduser()),
        "physics_fit": {k: v for k, v in fit.items() if k != "samples"},
        "n_params": int(n_params(10)),
        "generations": args.generations,
        "population": args.population,
        "frames": args.frames,
        "seeds": [{k: v for k, v in r.items() if k != "theta"} for r in results],
        "best_seed": results[0]["seed"],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"\nsaved {out} (seed {results[0]['seed']}, {results[0]['laps']:.2f} laps) "
          f"and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
