"""The slot-in driver. Load a policy, hand it an env, let it drive.

    from src.mk64.env import MK64Env
    from src.policy.drive import Driver

    with MK64Env() as env:
        d = Driver.load("policies/es_v2.npy")
        result = d.drive(env, "states/luigi_raceway_tt_start.st")
        print(result["completed_lap"], result["frames"])

The policy is 385 numbers. It takes the ten-number observation and returns a
steering value; the throttle is held open. Nothing in it knows what Mario Kart
is, which is the point: give it another world that emits the same ten numbers
and it drives that instead.
"""

import json
from pathlib import Path

import numpy as np

from ..mk64 import player
from ..obs.mk64_adapter import MK64Adapter
from ..obs.track import OBS_FIELDS, OBS_VERSION
from ..sim.train_es import act

ROOT = Path(__file__).resolve().parents[2]


class Driver:
    """A trained driving policy, plus the harness to run it."""

    def __init__(self, theta, obs_version=OBS_VERSION, name="es"):
        self.theta = np.asarray(theta, dtype=float).reshape(1, -1)
        self.obs_version = obs_version
        self.name = name

    @classmethod
    def load(cls, path, expect_obs_version=OBS_VERSION):
        path = Path(path)
        theta = np.load(path)
        meta_path = path.with_suffix(".json")
        version = OBS_VERSION
        if meta_path.exists():
            version = json.loads(meta_path.read_text()).get("obs_version", OBS_VERSION)
        if expect_obs_version is not None and version != expect_obs_version:
            raise ValueError(
                f"{path.name} was trained against observation v{version}, "
                f"caller has v{expect_obs_version}")
        return cls(theta, version, path.stem)

    def steer(self, obs):
        """Ten numbers in, one steering value in [-1, 1] out."""
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if obs.shape[1] != len(OBS_FIELDS):
            raise ValueError(f"expected {len(OBS_FIELDS)} inputs, got {obs.shape[1]}")
        return float(np.clip(act(self.theta, obs, obs.shape[1])[0], -1, 1))

    def drive(self, env, state_path, track_file="tracks/luigi_raceway.json",
              max_frames=4000, record=False):
        """Drive from the start line until a lap completes or frames run out."""
        ad = MK64Adapter(env, str(ROOT / track_file))
        obs = ad.reset(str(ROOT / state_path) if not Path(state_path).exists() else state_path)
        # -1 on the line, 0 once crossed; compare against 0 or the start of the
        # lap looks like the end of it.
        lap0 = max(0, player.read(env)["lapCount"])
        lats, sps, log = [], [], []
        for k in range(max_frames):
            s = self.steer(obs)
            env.step(1, accel=True, steer=s)
            st = player.read(env)
            obs = ad.observe(st)
            _, _, lat = ad.track.project(st["pos"], hint=ad.hint)
            lats.append(abs(lat)); sps.append(st["speed"])
            if record:
                log.append({"frame": env.frame, "steer": s, "speed": st["speed"],
                            "lateral": lat, "pos": st["pos"], "lap": st["lapCount"]})
            if st["lapCount"] > lap0:
                break
        lats = np.array(lats); sps = np.array(sps)
        res = {
            "policy": self.name,
            "completed_lap": bool(player.read(env)["lapCount"] > lap0),
            "frames": len(lats),
            "seconds": round(len(lats) / 60.0, 2),
            "mean_speed": float(sps.mean()), "max_speed": float(sps.max()),
            "mean_abs_lateral": float(lats.mean()),
            "frames_offroad": int((lats > 72).sum()),
        }
        if record:
            res["log"] = log
        return res
