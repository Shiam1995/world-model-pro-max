"""Lift MK64's kart physics constants out of the decomp into a JSON knob set.

`src/data/kart_attributes.c` is 430 lines of pure parameters — top speed per
engine class, a ten-point acceleration curve per character, friction, gravity,
bounding box. That is exactly the "physics becomes a parameter set" that SPEC.md
asks for in stage 3, already written down by Nintendo.

We take the numbers, not the code. A faithful port of player_controller.c (5,096
lines of mostly-unnamed functions, plus collision and surface handling) is a
different and much larger project; the constants plus a compact model that is
*calibrated against the emulator* gets the useful part at a fraction of the cost.
"""

import json
import re
from pathlib import Path

DECOMP = Path("/home/shiamjuniorchuttoo/Lets Burn Tokens/mk64")
SRC = DECOMP / "src/data/kart_attributes.c"

CHARACTERS = ["mario", "luigi", "yoshi", "toad", "dk", "wario", "peach", "bowser"]

WANTED = [
    "gTopSpeed50cc", "gTopSpeed100cc", "gTopSpeed150cc", "gTopSpeedExtra",
    "gTopSpeedBattle", "gKartFrictionTable", "gKartGravityTable",
    "gKartTopSpeedTable", "gKartBoundingBoxSizeTable",
    "gKartAccelerationMario", "gKartAccelerationLuigi", "gKartAccelerationYoshi",
    "gKartAccelerationToad", "gKartAccelerationDK", "gKartAccelerationWario",
    "gKartAccelerationPeach", "gKartAccelerationBowser",
]


def parse(text, name):
    m = re.search(rf"\b{name}\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not m:
        return None
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return [float(v) for v in re.findall(r"-?\d+\.?\d*", body)]


def main():
    text = SRC.read_text()
    out = {"source": str(SRC.relative_to(DECOMP)), "characters": CHARACTERS}
    for name in WANTED:
        v = parse(text, name)
        if v is None:
            print(f"  ! missing {name}")
        else:
            out[name] = v
    Path("tracks").mkdir(exist_ok=True)
    dest = Path("sim/kart_attributes.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    print(f"  Mario 100cc top speed : {out['gTopSpeed100cc'][0]}")
    print(f"  Mario accel curve     : {out['gKartAccelerationMario']}")
    print(f"  friction / gravity    : {out['gKartFrictionTable'][0]} / {out['gKartGravityTable'][0]}")


if __name__ == "__main__":
    main()
