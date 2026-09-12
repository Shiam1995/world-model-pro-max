"""Put a trained policy on the car in Unreal and drive the circuit.

    # the policy trained for this track and this vehicle
    python3 scripts/drive_unreal.py \
        --policy policies/kingscross_mk64_v1.npy \
        --track ~/kx/unreal_track_kingscross_figure8_mk64.json --kart --laps 1

    # the Mario Kart 64 policy, unchanged, on the same circuit
    python3 scripts/drive_unreal.py \
        --policy policies/luigi_raceway_v2.npy \
        --track ~/kx/unreal_track_kingscross_figure8_mk64.json --kart --laps 1

The policy is 385 numbers taking ten track-relative inputs and returning steering, with
the throttle held open. Nothing about it is Unreal-specific -- `UnrealAdapter` turns the
level's raw position and velocity into the same ten numbers the MK64 side emits, which is
the entire reason a policy can cross between them.

The length scale is derived from the track's lap length rather than set by hand, so the
policy sees a circuit with the same road-width-to-lap-length proportions it trained on.
See `src/obs/unreal_adapter` for why that, and not a physical metres-per-unit, is the
scale that matters.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UNREAL_T5 = Path(os.environ.get("UNREAL_T5", Path.home() / "unreal-t5"))
sys.path.insert(0, str(UNREAL_T5 / "env"))

from src.obs.unreal_adapter import UnrealAdapter, units_per_metre_for_lap  # noqa: E402
from src.policy.drive import Driver                                        # noqa: E402


def load_centreline(path, flip_y=True):
    """Centreline in Unreal world centimetres, matching what the level drives.

    flip_y mirrors Y to match `-trackflipy`: Unreal's OBJ importer negated Y when the
    world came in, so the level is a mirror of the authored frame and the policy has to
    be shown the world the car is actually in.
    """
    doc = json.loads(Path(path).expanduser().read_text())
    sign = -1.0 if flip_y else 1.0
    points = [[s["x"] * 100.0, s["y"] * 100.0 * sign, s["z"] * 100.0]
              for s in doc["centreline"]]
    return doc, points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="one step is 100 ms of simulated time")
    parser.add_argument("--port", type=int, default=9940)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--kart", action="store_true")
    parser.add_argument("--handedness", type=int, default=1, choices=(1, -1),
                        help="measured, not guessed: scripts/replay_openloop.py reports "
                             "it. Do not flip this by watching a policy fail.")
    parser.add_argument("--road-width-m", type=float, default=12.0)
    parser.add_argument("--units-per-metre", type=float, default=None,
                        help="override the length scale instead of deriving it from this "
                             "track's lap length. Use it to drive the full-scale circuit "
                             "at the scale the policy trained on: the observation carries "
                             "no absolute position and no lap counter, so a policy cannot "
                             "tell a 2392 m lap from a 1004 m one -- only road width and "
                             "corner curvature reach it, and holding units/m fixed holds "
                             "both of those where they were trained.")
    parser.add_argument("--throttle", type=float, default=1.0)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--vehicle-args", default="")
    args = parser.parse_args()

    from unreal_env import UnrealEnv

    doc, centreline = load_centreline(args.track)
    lap_m = float(doc["length_m"])
    units_per_metre = args.units_per_metre or units_per_metre_for_lap(lap_m)
    driver = Driver.load(args.policy)

    extra = ["-nofog"]
    if args.kart:
        extra.append("-kart")
    extra += [a for a in args.vehicle_args.split() if a]

    print(f"policy   {driver.name} ({driver.theta.size} parameters)")
    print(f"track    {doc['name']}: lap {lap_m:.1f} m, {len(centreline)} centreline points")
    print(f"scale    {units_per_metre:.3f} MK64 units/m, "
          f"half width {args.road_width_m / 2 * units_per_metre:.1f} units (MK64 Luigi is 72)")
    print(f"vehicle  {'kart preset' if args.kart else 'template sports car'}", flush=True)

    adapter = UnrealAdapter(
        centreline,
        up_axis="z",
        units_per_metre=100.0,
        half_width=args.road_width_m / 2.0 * 100.0,
        handedness=args.handedness,
        ref_units_per_metre=units_per_metre,
        seconds_per_step=0.1,
    )

    kwargs = dict(port=args.port, track=Path(args.track).expanduser(),
                  frames_dir=args.frames_dir if args.frames_dir is not None else None,
                  headless=not args.windowed, extra_args=extra)

    speeds, laterals, steers = [], [], []
    started = time.time()
    laps_done = 0
    reason = "step budget"
    streets = []

    with UnrealEnv(**kwargs) as env:
        env.reset()
        adapter.reset()
        telemetry = env.last_telemetry
        for step in range(args.max_steps):
            pos = [telemetry.get("x_cm", 0.0), telemetry.get("y_cm", 0.0),
                   telemetry.get("z_cm", 0.0)]
            vel = [telemetry.get("vx_cms", 0.0), telemetry.get("vy_cms", 0.0),
                   telemetry.get("vz_cms", 0.0)]
            obs = adapter.observe(pos, vel)
            steer = driver.steer(obs)
            env.step(args.throttle, steer)
            telemetry = env.last_telemetry

            speeds.append(float(telemetry.get("speed_kph", 0.0)))
            steers.append(steer)
            _, _, lateral = adapter.track.project(
                [p * adapter.scale for p in adapter._ground(pos)], hint=adapter.hint)
            laterals.append(abs(float(lateral)))
            street = telemetry.get("street")
            if street and (not streets or streets[-1] != street):
                streets.append(street)

            laps_done = int(telemetry.get("laps", 0))
            if laps_done >= args.laps:
                reason = "laps completed"
                break
            if step % 50 == 0:
                print(f"  step {step:4d} {telemetry.get('speed_kph', 0):5.1f} km/h  "
                      f"wp {telemetry.get('waypoint', 0):2}  "
                      f"lat {laterals[-1]:6.1f}u  steer {steer:+.2f}  "
                      f"{telemetry.get('street', '')}", flush=True)
        else:
            reason = "step budget"

    elapsed = time.time() - started
    half_width_units = args.road_width_m / 2.0 * units_per_metre
    result = {
        "policy": driver.name,
        "policy_file": args.policy,
        "track": doc["name"],
        "lap_m": lap_m,
        "units_per_metre": units_per_metre,
        "vehicle": "kart" if args.kart else "sports_car",
        "laps_completed": laps_done,
        "laps_requested": args.laps,
        "reason": reason,
        "steps": len(speeds),
        "sim_seconds": round(len(speeds) * 0.1, 2),
        "wall_seconds": round(elapsed, 1),
        "mean_speed_kph": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "max_speed_kph": round(float(np.max(speeds)), 2) if speeds else 0.0,
        "mean_abs_lateral_units": round(float(np.mean(laterals)), 1) if laterals else 0.0,
        "max_abs_lateral_units": round(float(np.max(laterals)), 1) if laterals else 0.0,
        "half_width_units": round(half_width_units, 1),
        "steps_off_road": int(sum(1 for v in laterals if v > half_width_units)),
        "mean_abs_steer": round(float(np.mean(np.abs(steers))), 3) if steers else 0.0,
        "streets": streets,
    }

    print()
    for key in ("laps_completed", "reason", "sim_seconds", "mean_speed_kph",
                "max_speed_kph", "mean_abs_lateral_units", "half_width_units",
                "steps_off_road", "mean_abs_steer"):
        print(f"  {key:26} {result[key]}")
    print(f"  streets reached            {len(streets)}: {' -> '.join(streets[:8])}"
          f"{' ...' if len(streets) > 8 else ''}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.report}")
    return result


if __name__ == "__main__":
    main()
