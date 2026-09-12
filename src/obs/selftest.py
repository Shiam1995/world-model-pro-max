"""Check the shared observation vector actually describes the world.

The vector is the project's spine: if it is subtly wrong, every policy trained
on it is wrong in the same way, in all four stages, and nothing will look
broken. So these assert meaning, not just shape.
"""

import math
import sys

import numpy as np

from ..mk64.env import MK64Env
from .mk64_adapter import MK64Adapter
from .track import OBS_FIELDS, OBS_VERSION

STATE = "states/luigi_raceway_tt_start.st"


def main():
    env = MK64Env()
    env.step(frames=20)
    ad = MK64Adapter(env)
    obs = ad.reset(STATE)

    assert obs.shape == (len(OBS_FIELDS),), f"bad shape {obs.shape}"
    assert np.all(np.isfinite(obs)), f"non-finite observation {obs}"
    print(f"[1] observation v{OBS_VERSION}: {len(OBS_FIELDS)} finite numbers")

    # The opening straight: a kart driving straight should read as on the line
    # and pointing down it. If this drifts, projection or heading is wrong.
    lat, err, prog = [], [], []
    for _ in range(6):
        ad.step(frames=40, accel=True)
        st = ad.raw()
        i, s, off = ad.track.project(st["pos"])
        o = ad.observe(st)
        lat.append(abs(off))
        err.append(abs(math.degrees(math.atan2(o[2], o[3]))))
        prog.append(s)
    assert max(lat) < 10, f"drifted off the centreline on a straight: {max(lat):.1f}"
    assert max(err) < 10, f"heading error on a straight: {max(err):.1f} deg"
    assert all(b > a for a, b in zip(prog, prog[1:])), f"progress not monotonic: {prog}"
    print(f"[2] straight: |lateral| <= {max(lat):.1f}, |heading err| <= {max(err):.1f} deg, "
          f"progress monotonic")

    # The corner preview has to LEAD. Far lookahead must see the first turn
    # while the near one still reads flat, or the policy learns to brake late
    # for ever.
    o = ad.observe()
    near, far = abs(o[4]), abs(o[7])
    assert far > near, f"far lookahead ({far:.2f}) should lead near ({near:.2f})"
    print(f"[3] corner preview leads: curv+2400 {far:.2f} vs curv+300 {near:.2f}")

    # Nothing in the vector may identify the track, or it cannot transfer.
    banned = {"x", "z", "pos", "lap", "course", "checkpoint"}
    assert not (banned & set(OBS_FIELDS)), f"observation leaks absolute state: {OBS_FIELDS}"
    print("[4] no absolute position, lap or course identity in the vector")

    env.close()
    print("\nOBSERVATION CHECKS PASSED")


if __name__ == "__main__":
    main()
