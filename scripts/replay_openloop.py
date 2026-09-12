"""Replay one fixed action script in the sim and in Unreal, and diff the trajectories.

The check the README puts above every other: **when sim-to-real fails, replay a fixed
action sequence in both worlds before touching the policy.** On the N64 side that single
run found a mirrored turn direction in minutes, after hours of closed-loop policy
evaluation had hidden it -- because a policy that fails can always be blamed on the
policy.

This does the same for Unreal. It takes the recording that `calibrate_unreal.py` already
made, replays exactly those actions through the fitted sim, and reports where the two
disagree -- in turn direction first, then speed, then the path itself.

    python3 scripts/replay_openloop.py runs/unreal_kart_calibration.jsonl \
        --physics runs/unreal_kart_physics.json --lap-m 1003.6

A pass here does not mean the sim is right about everything. It means the two worlds
agree about which way the vehicle turns and roughly how fast it goes, which is the part
whose disagreement is invisible from inside a training run.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.obs.track import Track                                          # noqa: E402
from src.obs.unreal_adapter import (MK64_FRAMES_PER_SECOND,              # noqa: E402
                                    MK64_TICKS_PER_FRAME,
                                    units_per_metre_for_lap)
from src.sim.kart import KartSim, Physics                                 # noqa: E402

TICKS_PER_SECOND = MK64_FRAMES_PER_SECOND * MK64_TICKS_PER_FRAME


def synthetic_track(units_per_metre):
    """A long oval, only so KartSim has something to project onto.

    Acceleration, yaw and sideslip are functions of speed and steering, so the shape does
    not matter -- but the track is not entirely inert, which cost a run to find out:
    `Physics.off_road` multiplies speed by 0.35 once a kart is beyond `half_width`, and a
    straight-line replay leaves any finite loop within seconds. The first version of this
    script reported the sim 65% slower than Unreal and an RMSE of half the top speed. It
    was not the acceleration fit; it was the replay driving across the grass.

    So the caller sets off_road = 1.0 as well. That is the honest setting for this
    comparison anyway: the calibration was recorded on one uniform ground surface, with
    no material change between the carriageway and the rest of it.
    """
    radius = 400.0 * units_per_metre
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    points = np.stack([radius * np.cos(angles),
                       np.zeros_like(angles),
                       radius * np.sin(angles)], axis=1)
    return Track.from_lap(points, spacing=4.5 * units_per_metre, smooth_window=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--physics", required=True)
    parser.add_argument("--lap-m", type=float, required=True)
    parser.add_argument("--seconds-per-step", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.trace) if line.strip()]
    fit = json.loads(Path(args.physics).read_text())
    units_per_metre = units_per_metre_for_lap(args.lap_m)
    to_units_per_tick = (units_per_metre / 100.0) / TICKS_PER_SECOND
    frames_per_step = int(round(args.seconds_per_step * MK64_FRAMES_PER_SECOND))

    physics = Physics.calibrated(
        top_speed=float(fit["top_speed"]),
        accel_table=tuple(map(tuple, fit["accel_table"])),
        turn_table=tuple(map(tuple, fit["turn_table"])),
        steer_response=tuple(map(tuple, fit["steer_response"])),
        half_width=6.0 * units_per_metre,
        # No off-road penalty: see synthetic_track. The trace was driven on open ground.
        off_road=1.0,
    )

    track = synthetic_track(units_per_metre)
    sim = KartSim(track, physics=physics, n=1, seed=0)
    sim.reset(jitter_lateral=0.0, jitter_yaw=0.0, start_cp=0)
    # Start from rest pointing along +x in the sim's ground plane, to match a recording
    # that starts from rest. Where it starts does not matter; the heading it starts at
    # does, because the comparison is of heading *change*.
    sim.xz[:] = 0.0
    sim.yaw[:] = math.pi / 2.0     # yaw is atan2(tangent_x, tangent_z) in this sim
    sim.speed[:] = 0.0

    sim_speed, sim_heading, sim_path = [], [], []
    for row in rows:
        for _ in range(frames_per_step):
            sim.step(np.array([row["steer"]]), throttle=np.array([row["throttle"]]))
        sim_speed.append(float(sim.speed[0]))
        sim_heading.append(float(sim.yaw[0]))
        sim_path.append((float(sim.xz[0, 0]), float(sim.xz[0, 1])))

    real_speed = [math.hypot(r["vel"][0], r["vel"][1]) * to_units_per_tick for r in rows]
    real_heading = []
    last = None
    for row in rows:
        vx, vy = row["vel"][0], row["vel"][1]
        if vx * vx + vy * vy > 1.0:
            last = math.atan2(vy, vx)
        real_heading.append(last if last is not None else 0.0)

    # --- turn direction, per burst -------------------------------------------
    print(f"{args.trace}: {len(rows)} steps, {frames_per_step} sim frames each\n")
    print("turn direction, per steering burst (heading change over the burst):")
    print(f"  {'steer':>7} {'unreal':>12} {'sim':>12}   verdict")
    bursts = {}
    for index, row in enumerate(rows):
        level = row["steer"]
        if level == 0.0:
            continue
        bursts.setdefault((level, row["label"]), []).append(index)

    mismatches = 0
    for (level, label), indices in sorted(bursts.items()):
        if len(indices) < 3:
            continue
        first, final = indices[1], indices[-1]
        real_d = math.degrees(math.atan2(
            math.sin(real_heading[final] - real_heading[first]),
            math.cos(real_heading[final] - real_heading[first])))
        sim_d = math.degrees(math.atan2(
            math.sin(sim_heading[final] - sim_heading[first]),
            math.cos(sim_heading[final] - sim_heading[first])))
        # The sim's yaw is measured the other way round the circle from atan2(vy, vx):
        # what has to agree is the SIGN relative to the stick, not the absolute frame.
        agree = (real_d >= 0) == (sim_d <= 0)
        if not agree:
            mismatches += 1
        print(f"  {level:+7.2f} {real_d:+11.1f}d {sim_d:+11.1f}d   "
              f"{'ok' if agree else 'MIRRORED'}   [{label}]")

    # --- speed ---------------------------------------------------------------
    straight = [i for i, r in enumerate(rows)
                if r["steer"] == 0.0 and r["throttle"] >= 0.99]
    if straight:
        errors = [sim_speed[i] - real_speed[i] for i in straight]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        print(f"\nlongitudinal (full throttle, straight, {len(straight)} steps):")
        print(f"  top speed   sim {max(sim_speed):.3f} vs unreal {max(real_speed):.3f} "
              f"units/tick  ({100 * (max(sim_speed) - max(real_speed)) / max(real_speed):+.1f}%)")
        print(f"  RMSE        {rmse:.4f} units/tick "
              f"({100 * rmse / max(real_speed):.1f}% of top speed)")

    verdict = "PASS" if mismatches == 0 else f"FAIL - {mismatches} burst(s) mirrored"
    print(f"\nturn direction: {verdict}")
    if mismatches:
        print("  Do not train against this sim. Flip the steering sign, or the adapter's")
        print("  handedness, and re-run this before anything else.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "steps": len(rows),
            "mirrored_bursts": mismatches,
            "sim_top_speed": max(sim_speed),
            "unreal_top_speed": max(real_speed),
            "sim_path": sim_path[::10],
        }, indent=1))
        print(f"wrote {args.out}")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
