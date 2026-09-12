"""Pull a course centreline out of the mk64 decomp's course_data.c.

MK64 ships the path its own CPU racers follow — `d_course_<name>_track_path`,
an array of TrackPathPoint {s16 x, y, z; u16 trackSectionId}. That is a real
centreline, authored by Nintendo, at 20-unit spacing.

Why use it instead of a lap driven by search (src/mk64/ghost.py): the search
driver could not finish a lap. Greedy "cover the most ground this half second"
has no way to tell a corner from a circle, and it wedged the kart against a wall
at segment 30 with the speed at 0.12. The game's own `nearestPathPointId` would
have fixed that, but it stays 0 for the player in a time trial.

So: where a world already knows its own road, read the road. That is the same
deal stages 3 and 4 strike anyway — Unreal gets the geometry from the decomp,
London gets it from OSM. Driving a lap blind is only necessary where no map
exists at all, and the ghost driver stays in the tree for exactly that case.
"""

import argparse
import json
import re
from pathlib import Path

DECOMP = Path("/home/shiamjuniorchuttoo/Lets Burn Tokens/mk64")
ROW = re.compile(r"\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*\}")


def extract(course, array="track_path"):
    src = DECOMP / "courses" / course / "course_data.c"
    text = src.read_text()
    name = f"d_course_{course}_{array}[]"
    start = text.index(name)
    end = text.index("};", start)
    pts = [(int(x), int(y), int(z), int(sec))
           for x, y, z, sec in ROW.findall(text[start:end])]
    # The array is terminated by a sentinel row of -32768s. Leaving it in puts
    # a point 32k units away and wrecks every arc length and projection.
    pts = [p for p in pts if -32768 not in p[:3]]
    if not pts:
        raise SystemExit(f"no points parsed from {name}")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("course", nargs="?", default="luigi_raceway")
    ap.add_argument("--array", default="track_path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pts = extract(a.course, a.array)
    out = Path(a.out or f"tracks/{a.course}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "course": a.course,
        "source": f"mk64 decomp d_course_{a.course}_{a.array}",
        "points": [[p[0], p[1], p[2]] for p in pts],
        "section": [p[3] for p in pts],
    }))
    xs = [p[0] for p in pts]; zs = [p[2] for p in pts]
    print(f"{a.course}: {len(pts)} points -> {out}")
    print(f"  x {min(xs)}..{max(xs)}   z {min(zs)}..{max(zs)}   "
          f"sections {len(set(p[3] for p in pts))}")


if __name__ == "__main__":
    main()
