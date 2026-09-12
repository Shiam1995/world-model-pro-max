"""Fit the sim to the emulator, and say how well it fits.

A simulator nobody compared against the real thing is a story. This collects
short, scripted traces from the actual ROM — accelerate from rest; hold a
steering angle at speed — and fits the handful of parameters that are not
decomp constants, then reports the residual so the error is a number rather
than a vibe.

Calibrated here:
  top_speed, accel_scale     from the throttle trace
  turn_rate, turn_falloff    from the steering traces
"""

import json
from pathlib import Path

import numpy as np

STATE = "states/luigi_raceway_tt_start.st"


def collect(env, frames=360, steers=(0.0, 0.5, 1.0)):
    """Scripted traces from the emulator: speed and heading, per frame."""
    from ..mk64 import player
    traces = {}
    for steer in steers:
        env.step(frames=10)
        env.load_state_file(STATE)
        sp, yaw, pos = [], [], []
        for _ in range(frames):
            env.step(1, accel=True, steer=steer)
            st = player.read(env)
            sp.append(st["speed"])
            yaw.append(st["yaw"])
            pos.append(st["pos"])
        traces[str(steer)] = {"speed": sp, "yaw": yaw, "pos": pos}
    return traces


def _sim_speed(top, scale, curve, n, ticks=2, drag=0.002):
    s, out = 0.0, []
    for _ in range(n):
        for _ in range(ticks):
            frac = min(max(s / top, 0.0), 0.999)
            a = curve[int(frac * len(curve))] * scale
            s = min(s + a - drag * s, top)
        out.append(s)
    return np.array(out)


def fit_speed(trace, curve, tops=None, scales=None):
    """Grid-search top speed and acceleration scale against the real trace."""
    real = np.array(trace["speed"], dtype=float)
    tops = tops if tops is not None else np.linspace(5.0, 6.5, 61)
    scales = scales if scales is not None else np.linspace(0.004, 0.060, 141)
    best = (None, None, 1e18)
    for top in tops:
        for sc in scales:
            pred = _sim_speed(top, sc, curve, len(real))
            err = float(np.sqrt(np.mean((pred - real) ** 2)))
            if err < best[2]:
                best = (top, sc, err)
    return best


def fit_turn(traces, top_speed, curve, accel_scale):
    """Fit steering authority from how fast heading changes at a given speed."""
    rows = []
    for key, tr in traces.items():
        steer = float(key)
        if steer == 0.0:
            continue
        yaw = np.unwrap(np.array(tr["yaw"], dtype=float))
        sp = np.array(tr["speed"], dtype=float)
        dyaw = np.diff(yaw)
        # Use the stretch where the kart is up to speed and still on the road.
        lo = max(10, int(len(dyaw) * 0.15))
        hi = min(len(dyaw), int(len(dyaw) * 0.55))
        for k in range(lo, hi):
            rows.append((steer, sp[k], abs(dyaw[k])))
    rows = np.array(rows)
    if not len(rows):
        return 0.055, 0.55, float("nan")

    def predict(rate, falloff):
        steer, sp, _ = rows[:, 0], rows[:, 1], rows[:, 2]
        auth = np.clip(1.0 - falloff * (sp / top_speed), 0.05, 1.0)
        ramp = np.clip(sp / (0.25 * top_speed), 0, 1)
        return np.abs(steer) * rate * auth * ramp / 2.0   # per frame -> per tick x2

    best = (0.055, 0.55, 1e18)
    for rate in np.linspace(0.005, 0.12, 116):
        for falloff in np.linspace(0.0, 0.95, 20):
            err = float(np.sqrt(np.mean((predict(rate, falloff) - rows[:, 2]) ** 2)))
            if err < best[2]:
                best = (rate, falloff, err)
    return best


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.mk64.env import MK64Env
    from src.sim.kart import Physics

    env = MK64Env()
    print("collecting emulator traces", flush=True)
    traces = collect(env)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/calibration_traces.json").write_text(json.dumps(traces))
    env.close()

    curve = Physics.from_decomp().accel_curve
    top, scale, serr = fit_speed(traces["0.0"], curve)
    rate, falloff, terr = fit_turn(traces, top, curve, scale)

    real = np.array(traces["0.0"]["speed"])
    pred = _sim_speed(top, scale, curve, len(real))
    out = {
        "top_speed": float(top), "accel_scale": float(scale),
        "turn_rate": float(rate), "turn_speed_falloff": float(falloff),
        "speed_rmse": float(serr), "turn_rmse_rad": float(terr),
        "real_top_speed": float(real.max()), "sim_top_speed": float(pred.max()),
    }
    Path("sim/calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nspeed trace RMSE {serr:.4f} units/tick "
          f"({100*serr/max(real.max(),1e-9):.1f}% of top speed)")
    print(f"turn rate RMSE   {terr:.5f} rad/frame")


if __name__ == "__main__":
    main()
