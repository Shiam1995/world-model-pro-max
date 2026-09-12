"""Record one open-loop calibration run in Unreal, and read the handedness off it.

This is the script `src/obs/unreal_adapter.py` tells you to trust instead of watching a
policy fail. It drives a *fixed* action script -- no controller, no policy -- and logs
position and velocity every step. From that one recording come the four things a sim of
this vehicle needs: which way `+steer` actually turns it, the acceleration curve, the top
speed, and the steering response including any deadzone.

    python3 scripts/calibrate_unreal.py --out runs/unreal_kx_calibration.jsonl

The lesson behind it (README, "the one that matters most"): when sim-to-real failed on
the N64 side, the cause was a mirrored turn direction, and hours of closed-loop policy
evaluation had hidden it -- because a policy that fails can always be blamed on the
policy. An open-loop replay found it in a single run.

Steering is measured on open ground, not on the circuit. Holding full lock for two
seconds on a 12 m street just puts the car into the buildings, and a collision in the
middle of a calibration trace is indistinguishable from a vehicle that cannot turn.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UNREAL_T5 = Path(os.environ.get("UNREAL_T5", Path.home() / "unreal-t5"))
sys.path.insert(0, str(UNREAL_T5 / "env"))


#: Part throttles used to hold the vehicle at a range of steady speeds before turning.
#: Anything with drag reaches a different equilibrium speed for each throttle setting,
#: which is the cheapest way to get a turn measurement at low speed without braking.
HOLD_THROTTLES = (0.22, 0.40, 0.65)


def action_script(steer_levels, straight_steps, turn_steps, settle_steps):
    """(label, throttle, steer) per step, in three phases.

    A. Full throttle from rest, straight. This is the only phase the acceleration fit
       reads, so nothing is allowed to disturb it.
    B. Each steer level held at racing speed, separated by straights. This gives the
       steering response curve -- how much of full lock each stick position buys.
    C. Full lock again at a range of *lower* speeds, reached by holding part throttle
       until the vehicle settles.

    Phase C exists because without it every turn measurement lands in the top speed bin.
    The first version of this script accelerated for seven seconds before its first
    burst, and produced a turn table with two bins, both above 80% of top speed -- so the
    sim had no idea how the vehicle steers slowly and would have invented it. That
    invention is named in the README as the single thing that broke sim-to-real on the
    N64 side: it let a policy carry a large constant steering bias for free, and the
    bias drove the real kart off the road in the first two seconds.
    """
    plan = [("launch", 1.0, 0.0)] * straight_steps
    for level in steer_levels:
        plan += [(f"steer{level:+.2f}", 1.0, level)] * turn_steps
        plan += [("settle", 1.0, 0.0)] * settle_steps

    full = max(steer_levels, key=abs) if steer_levels else 1.0
    for throttle in HOLD_THROTTLES:
        # Hold long enough to actually reach equilibrium, not just to slow down.
        plan += [(f"hold{throttle:.2f}", throttle, 0.0)] * (settle_steps * 3)
        plan += [(f"slowturn{throttle:.2f}", throttle, full)] * turn_steps
    plan += [("coast", 0.0, 0.0)] * settle_steps
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "runs" / "unreal_kx_calibration.jsonl"))
    parser.add_argument("--port", type=int, default=9931)
    parser.add_argument("--straight-steps", type=int, default=70,
                        help="steps at zero steer from rest; 1 step = 100 ms")
    parser.add_argument("--turn-steps", type=int, default=22)
    parser.add_argument("--settle-steps", type=int, default=16)
    parser.add_argument("--levels", default="0.15,0.30,0.50,0.80,1.00,-0.80",
                        help="steer levels to hold, in order. The small ones are there to "
                             "find a deadzone; the negative one checks the response is "
                             "symmetric rather than assuming it.")
    parser.add_argument("--level", default=None,
                        help="map to drive; default is the open Westminster ground")
    parser.add_argument("--nobuildings", action="store_true", default=True)
    parser.add_argument("--windowed", action="store_true",
                        help="show the window instead of rendering offscreen")
    parser.add_argument("--kart", action="store_true",
                        help="calibrate the kart preset rather than the template sports "
                             "car. Worth being deliberate about: the sports car does "
                             "235 km/h, which no policy will drive round a 12 m street "
                             "circuit, so the kart is the vehicle that actually races.")
    parser.add_argument("--vehicle-args", default="",
                        help="extra dial arguments, e.g. '-torquescale=0.4 -dragcoeff=2'")
    args = parser.parse_args()

    from unreal_env import UnrealEnv

    levels = [float(v) for v in args.levels.split(",")]
    plan = action_script(levels, args.straight_steps, args.turn_steps, args.settle_steps)

    extra = ["-nofog"]
    if args.kart:
        extra.append("-kart")
    extra += [a for a in args.vehicle_args.split() if a]
    if args.nobuildings:
        # Open ground: the generated Westminster level with its kerbside blocks removed is
        # the only flat, obstacle-free space either map has. The vehicle's dynamics do not
        # depend on which map it is on, but whether it hits a wall very much does.
        extra.append("-nobuildings")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(port=args.port, frames_dir="", headless=not args.windowed,
                  extra_args=extra)
    if args.level:
        kwargs["level"] = args.level

    print(f"calibration: {len(plan)} steps ({len(plan) * 0.1:.1f} s of sim), "
          f"levels {levels}", flush=True)

    rows = []
    with UnrealEnv(**kwargs) as env:
        env.reset()
        for index, (label, throttle, steer) in enumerate(plan):
            env.step(throttle, steer)
            telemetry = env.last_telemetry
            row = {
                "frame": index,
                "label": label,
                "throttle": throttle,
                "steer": steer,
                "pos": [telemetry.get("x_cm", 0.0), telemetry.get("y_cm", 0.0),
                        telemetry.get("z_cm", 0.0)],
                "vel": [telemetry.get("vx_cms", 0.0), telemetry.get("vy_cms", 0.0),
                        telemetry.get("vz_cms", 0.0)],
                "speed_kph": telemetry.get("speed_kph", 0.0),
                "yaw_deg": telemetry.get("heading", 0.0),
            }
            rows.append(row)
            if index % 25 == 0:
                print(f"  step {index:3d} {label:10} {row['speed_kph']:6.1f} km/h "
                      f"at ({row['pos'][0] / 100:7.1f}, {row['pos'][1] / 100:7.1f})", flush=True)

    with open(out_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")

    report_handedness(rows)


def report_handedness(rows):
    """The single question this run exists to answer, stated as a number.

    Yaw rate is taken from the velocity vector rather than the reported yaw angle, for the
    same reason the adapter asks for a vector: an angle needs a sign convention agreed
    with whoever wrote the other end, and that agreement is exactly what was wrong before.
    """
    def heading(row):
        vx, vy = row["vel"][0], row["vel"][1]
        return math.atan2(vy, vx) if (vx * vx + vy * vy) > 1.0 else None

    for level in (0.80, -0.80):
        burst = [r for r in rows if r["steer"] == level]
        if len(burst) < 3:
            continue
        first, last = heading(burst[1]), heading(burst[-1])
        if first is None or last is None:
            print(f"  steer {level:+.2f}: car was not moving, cannot tell")
            continue
        delta = math.degrees(math.atan2(math.sin(last - first), math.cos(last - first)))
        seconds = (len(burst) - 1) * 0.1
        print(f"  steer {level:+.2f}: heading changed {delta:+7.1f} deg over {seconds:.1f}s "
              f"({delta / seconds:+6.1f} deg/s)")

    print("  -> +steer turning LEFT (negative yaw) means handedness = -1 for UnrealAdapter")


if __name__ == "__main__":
    main()
