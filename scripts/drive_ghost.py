import sys, json, time
sys.path.insert(0, '/home/shiamjuniorchuttoo/Projects/Mario to sonic')
from pathlib import Path
from src.mk64.env import MK64Env
from src.mk64.ghost import drive_lap

t0 = time.time()
env = MK64Env()
log = drive_lap(env, "states/luigi_raceway_tt_start.st")
out = Path("runs/ghost_luigi_raceway.json")
out.write_text(json.dumps(log))
print(f"\nwrote {out} — {len(log)} frames in {time.time()-t0:.0f}s")
env.close()
