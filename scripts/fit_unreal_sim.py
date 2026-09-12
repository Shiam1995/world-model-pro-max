"""Fit the kart sim to a vehicle measured in Unreal, from one open-loop trace.

Input is the jsonl that `scripts/calibrate_unreal.py` records; output is a Physics knob
set in MK64 units that `src/sim/kart.py` can be trained against. Nothing here is guessed:
each field is measured off the trace, the same way `src/sim/calibrate.py` fits the sim to
the emulator.

    python3 scripts/fit_unreal_sim.py runs/unreal_kx_calibration.jsonl \
        --lap-m 1003.6 --out runs/unreal_kx_physics.json

Why the tables and not the scalar knobs. `Physics` carries both: `turn_rate` with a
falloff, and a measured `turn_table`. The README is explicit that the invented low-speed
ramp behind the scalar form is "the single thing that broke sim-to-real", because it let a
policy carry a large constant steering bias for free. Measured tables replace it, so this
fits tables and leaves the scalars alone.

Units. The trace is Unreal centimetres and seconds; the sim is MK64 length units per tick
(speed, acceleration) and per frame (yaw). `--lap-m` sets the length scale the same way
`UnrealAdapter` does -- by mapping this track's lap onto Luigi Raceway's -- so the sim, the
adapter and the policy all agree. Getting that wrong does not error, it just trains a
policy for a vehicle that does not exist.
"""

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.obs.unreal_adapter import (  # noqa: E402
    MK64_FRAMES_PER_SECOND, MK64_TICKS_PER_FRAME, units_per_metre_for_lap)

TICKS_PER_SECOND = MK64_FRAMES_PER_SECOND * MK64_TICKS_PER_FRAME


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if len(rows) < 20:
        raise SystemExit(f"{path}: only {len(rows)} rows, that is not a calibration run")
    return rows


def planar_speed_cms(row):
    vx, vy = row["vel"][0], row["vel"][1]
    return math.hypot(vx, vy)


def heading(row):
    vx, vy = row["vel"][0], row["vel"][1]
    return math.atan2(vy, vx) if (vx * vx + vy * vy) > 1.0 else None


def angle_delta(a, b):
    """Smallest signed b - a."""
    return math.atan2(math.sin(b - a), math.cos(b - a))


def fit(rows, units_per_metre, seconds_per_step):
    """Everything the sim needs, in MK64 units."""
    # cm/s -> MK64 units per tick
    to_units_per_tick = (units_per_metre / 100.0) / TICKS_PER_SECOND
    frames_per_step = seconds_per_step * MK64_FRAMES_PER_SECOND

    speeds = [planar_speed_cms(r) * to_units_per_tick for r in rows]
    top_speed = max(speeds)

    # --- acceleration: a(v) per frame, off the full-throttle straight from rest ---
    accel_samples = []
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        if current["steer"] != 0.0 or current["throttle"] < 0.99:
            continue
        if previous["steer"] != 0.0 or previous["throttle"] < 0.99:
            continue
        dv = speeds[index] - speeds[index - 1]
        accel_samples.append((speeds[index - 1], dv / frames_per_step))

    accel_table = bucket(accel_samples, top_speed, buckets=10)

    # --- steering: |dyaw| per frame at each level held ---
    by_level = {}
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        level = current["steer"]
        if level == 0.0 or previous["steer"] != level:
            continue      # skip the first step of a burst: the input has just changed
        before, after = heading(previous), heading(current)
        if before is None or after is None:
            continue
        rate = angle_delta(before, after) / frames_per_step
        by_level.setdefault(level, []).append((speeds[index - 1], rate))

    # Handedness: does a positive stick produce a positive yaw in this frame?
    signed = [r for level, samples in by_level.items() if level > 0 for _, r in samples]
    mean_positive_yaw = sum(signed) / len(signed) if signed else 0.0

    # Full lock defines the turn table; everything else is a fraction of it.
    full_level = max(by_level, key=lambda k: abs(k)) if by_level else None
    turn_table = []
    coverage = None
    if full_level is not None:
        full_samples = [(v, abs(r)) for v, r in by_level[full_level]]
        turn_table = bucket(full_samples, top_speed, buckets=8)
        # Say how much of the speed range was actually measured. A table that only covers
        # the top of it means the sim will invent low-speed steering, and an invented
        # low-speed steering ramp is what broke sim-to-real once already.
        if full_samples:
            lowest = min(v for v, _ in full_samples)
            coverage = {
                "lowest_measured_speed": round(lowest, 4),
                "fraction_of_top_speed": round(lowest / top_speed, 4) if top_speed else None,
                "bins": len(turn_table),
            }

    # --- steer response: stick -> fraction of full lock, at comparable speed ---
    response = [(0.0, 0.0)]
    full_rate = mean_abs([r for _, r in by_level.get(full_level, [])]) if full_level else 0.0
    for level in sorted(k for k in by_level if k > 0):
        rate = mean_abs([r for _, r in by_level[level]])
        response.append((abs(level), (rate / full_rate) if full_rate else 0.0))
    if response[-1][0] < 1.0:
        response.append((1.0, 1.0))

    return {
        "units_per_metre": units_per_metre,
        "seconds_per_step": seconds_per_step,
        "top_speed": top_speed,
        "accel_table": accel_table,
        "turn_table": turn_table,
        "steer_response": response,
        "handedness": 1 if mean_positive_yaw >= 0 else -1,
        "mean_yaw_at_positive_stick": mean_positive_yaw,
        "full_lock_level": abs(full_level) if full_level is not None else None,
        "turn_coverage": coverage,
        "samples": {"accel": len(accel_samples),
                    "steer": {str(k): len(v) for k, v in sorted(by_level.items())}},
    }


def mean_abs(values):
    return sum(abs(v) for v in values) / len(values) if values else 0.0


def bucket(samples, top, buckets):
    """Average y within `buckets` speed bins, returned as a sorted table."""
    if not samples or top <= 0:
        return []
    out = []
    for index in range(buckets):
        low = top * index / buckets
        high = top * (index + 1) / buckets
        inside = [y for x, y in samples if low <= x < high or (index == buckets - 1 and x >= low)]
        if inside:
            out.append((round((low + high) / 2, 6), round(sum(inside) / len(inside), 6)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--lap-m", type=float, required=True,
                        help="lap length of the track this policy will drive, in metres")
    parser.add_argument("--seconds-per-step", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load(args.trace)
    units_per_metre = units_per_metre_for_lap(args.lap_m)
    fitted = fit(rows, units_per_metre, args.seconds_per_step)

    print(f"{args.trace}: {len(rows)} rows")
    print(f"length scale      {units_per_metre:.3f} MK64 units/m (lap {args.lap_m:.1f} m)")
    print(f"top speed         {fitted['top_speed']:.3f} units/tick "
          f"(MK64 Mario 100cc is 5.6)")
    print(f"handedness        {fitted['handedness']:+d}  "
          f"(mean yaw at +stick {fitted['mean_yaw_at_positive_stick']:+.5f} rad/frame)")
    print(f"accel table       {len(fitted['accel_table'])} bins from "
          f"{fitted['samples']['accel']} samples")
    for v, a in fitted["accel_table"]:
        print(f"    v={v:6.3f}  a={a:+.5f} /frame")
    print(f"turn table        full lock = {fitted['full_lock_level']}")
    for v, r in fitted["turn_table"]:
        print(f"    v={v:6.3f}  |dyaw|={r:.5f} rad/frame")
    cover = fitted.get("turn_coverage")
    if cover:
        print(f"    measured down to v={cover['lowest_measured_speed']:.3f} "
              f"({cover['fraction_of_top_speed'] * 100:.0f}% of top speed) "
              f"in {cover['bins']} bins")
        if cover["fraction_of_top_speed"] and cover["fraction_of_top_speed"] > 0.5:
            print("    WARNING: no turn data below half top speed — the sim will have to "
                  "invent low-speed steering, which is the bug that broke transfer once")
    print("steer response    stick -> fraction of full lock")
    for stick, frac in fitted["steer_response"]:
        print(f"    {stick:4.2f} -> {frac:5.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(fitted, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
