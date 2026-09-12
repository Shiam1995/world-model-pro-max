# Mario to Sonic — handoff

**State: working.** A driving policy trained for 112 seconds in simulation, which
never saw the emulator, completes a **26.72 s lap of Luigi Raceway on the real
Nintendo 64**. Repeats bit-identically 3/3. 4 of 5 seeds transfer.

Repo: `~/Projects/Mario to sonic` (git, all work committed).
Demo page: https://claude.ai/code/artifact/090c40b4-cc74-4bd6-9343-d80eb31c7917

---

## 1. Run it in 60 seconds

```bash
cd ~/Projects/"Mario to sonic"
make                 # build the input plugin (needs gcc; ~2s)
make test            # harness: boots, steps, rollback is exact        (~40s)
make test-player     # kart telemetry is being read correctly          (~60s)
make test-obs        # the observation vector means what it says       (~60s)
```

Watch the AI drive the real game:

```bash
python3 scripts/demo_live.py policy      # emulator window, 60fps
```

Use the policy from code:

```python
from src.mk64.env import MK64Env
from src.policy.drive import Driver

with MK64Env() as env:
    d = Driver.load("policies/luigi_raceway_v2.npy")
    print(d.drive(env, "states/luigi_raceway_tt_start.st"))
# {'completed_lap': True, 'seconds': 26.72, 'mean_speed': 3.78, 'frames_offroad': 118}
```

---

## 2. What this is

Four stages, from `SPEC.md`:

1. **Solve Mario Kart 64** — done.
2. **Show confidence while driving** — not built (see §6).
3. **Courses into Unreal, physics on a knob** — the knob exists; the Unreal
   adapter exists and is tested; the map is external.
4. **Policies onto a random London track** — adapter ready, geometry not built.

**The spine.** Every world emits the same ten-number, track-relative observation:

```
speed, lateral_offset, heading_sin, heading_cos,
curv_300, curv_600, curv_1200, curv_2400, grade, progress_delta
```

It deliberately carries **no absolute position, no lap counter, no course
identity**. A policy that cannot tell Luigi Raceway from a London roundabout is
one that can be moved between them. `make test-obs` asserts this.

The policy is 385 numbers: an MLP 10→32→1, tanh, steering only, throttle held
open.

---

## 3. Plugging in Unreal

Server: `python3 scripts/unreal_server.py --port 8791` (line-delimited JSON).

**Once, on connect:**
```json
{"type":"track","centreline":[[x,y,z],...],"half_width":350.0,
 "up_axis":"z","units_per_metre":100.0}
```
Centreline runs along the middle of the road in driving order, closed loop,
points every 2–5 m.

**Every frame:**
```json
{"type":"state","pos":[x,y,z],"vel":[x,y,z]}
  ->  {"steer":-1..1,"throttle":0..1,"brake":0..1}
```

Send the **velocity vector, not a heading angle**. A vector needs no handedness,
zero-direction or sign agreed about it — and getting one of those backwards is
exactly the bug that cost a day here (§5).

**Also record one open-loop calibration run** (`scripts/record_unreal_calibration.md`):
drive a *fixed* script — 90 frames straight, 60 at +0.8, 60 straight, 60 at −0.8,
90 straight — logging `pos`/`vel` per frame. That single recording yields
handedness, acceleration curve, top speed and the steering response including
any deadzone.

**Expectation:** `luigi_raceway_v2` will connect and steer, but will not drive
your map well — it is tuned to Luigi Raceway's corners and MK64's handling. With
the track plus the calibration run, a policy native to your map trains in about
two minutes. A new world is a retrain, not a project.

---

## 4. How it works

**The emulator is the oracle, not the training loop.** `libmupen64plus.so.2` is
loaded into the Python process with ctypes, so `save_state()` is a cheap
`env.clone()` and replay is bit-exact. It runs ~100 fps on one core.

**The sim does the volume.** 493,000 kart-frames/second — about 5,000×. Physics
constants are lifted from the decompilation's `kart_attributes.c` (top speed per
engine class, a ten-point acceleration curve, friction, gravity), and everything
that is not a decomp constant is **calibrated against emulator traces**:

```
longitudinal RMSE 0.40 units/tick   (7.2% of top speed)
top speed  5.539 sim vs 5.581 real  (0.8%)
frames to 90% top: 133 sim vs 137 real
```

**Training** is evolution strategies — the sim's parallel karts *are* the
population, so one rollout evaluates a whole generation. 80 generations, 384
population, ~112 s.

**The map** comes from the decompilation's own CPU-racer path
(`d_course_<name>_track_path`). The kart's start position projects 0.5 units onto
the extracted centreline, which is the check that map and game agree.

---

## 5. Traps — read before debugging anything

Every one of these presents as "the agent learned nothing" rather than as an
error. Full detail in `README.md`.

| # | Trap |
|---|---|
| 1 | **Plugin attach order must be GFX → AUDIO → INPUT → RSP.** Otherwise the RSP binds to `dummy_gfx`, MK64 renders one framebuffer forever and never polls the controller. The core only *warns*. |
| 2 | **No true headless** — a dummy video plugin wedges MK64. The RDP has to be real. |
| 3 | **`GbCameraVideoCaptureBackend1` defaults to `"opencv"`** and hangs ROM startup on a camera that isn't there. Hangs the stock binary too. Set it to `""`. |
| 4 | **RDRAM is byte-swapped within each 32-bit word.** 32-bit reads work little-endian; every **16-bit** field needs `offset ^ 2` (8-bit would be `^ 3`). |
| 5 | **`M64P_DBG_PTR_RDRAM` is 1, not 4** (4 is `VI_REG`). Using 4 reads megabytes past a small allocation — heap-dependent crashes, and *tests that pass against the wrong memory*. |
| 6 | **`RandomizeInterrupt` defaults on** and makes replay non-reproducible by ~100 bytes. Turn it off. |
| 7 | **MK64 runs two physics ticks per frame.** `speed` and `pos−oldPos` are per tick; the kart covers `2 × speed` per frame. |
| 8 | **`lapCount` is −1 on the start line**, 0 after crossing. Compare against 0 or the *start* of the lap looks like the end. |
| 9 | **MK64 confirms every menu selection with its own "OK ?"** — five A presses from PLAYER SELECT to the track, not four. |
| 10 | **`SAVECOMPLETE` fires before the bytes reach disk** — the file is a truncated gzip. Wait until it decompresses. |
| 11 | **Steering is non-linear with a deadzone**: stick 0.15 gives *exactly zero* yaw rate, 0.30 gives 5% of full lock, saturating by 0.8. |

### The one that matters most

**When sim-to-real fails, replay a fixed action sequence in both worlds before
touching the policy.**

The transfer bug was that the sim's turn direction was *mirrored*: `+0.8` steer
moved the real kart toward `+x` and the sim kart toward `−x`. An open-loop replay
found it in a single run — 31 units apart while driving straight, 678 after two
turns. Hours of closed-loop policy evaluation had hidden it, because **a policy
that fails can always be blamed on the policy**. Ruled out by measurement along
the way: steering sign at the policy output, the yaw trig convention, and
parameter error in general (domain randomisation did not help, which is what
said the gap was systematic).

---

## 6. What is *not* built

- **Confidence (stage 2).** The ensemble machinery exists (`src/policy/model.py`)
  and four independently seeded policies are trained, one of which fails on the
  real console — so the data to calibrate disagreement against outcomes is
  there. The calibration itself is not done. This is the biggest missing
  headline feature.
- **Walls.** The sim models an off-road slowdown, not barriers, so it cannot
  represent a crash. `half_width = 72` is a single measurement off one corner.
- **London geometry.** No OSM data fetched. Needs one download (the project's
  sanctioned external fetch).
- **Rollback search on the real console** reaches 20% of a lap then wedges.
  Superseded by the trained policy; `src/mk64/search.py` is kept because it
  needs no model at all.
- **Generalisation across courses** — 21 course centrelines are extracted and a
  study is running (`scripts/generalise.py`, `runs/generalisation.json`). Until
  it finishes, *whether a policy drives an unseen track is unmeasured*, and that
  is the same question as whether it will drive the Unreal map.

---

## 7. File map

```
SPEC.md                     the four stages and the shared observation vector
README.md                   findings, at length
HANDOFF.md                  this file

src/mk64/env.py             MK64Env: step / save_state / load_state / ram
src/mk64/core.py            ctypes binding to libmupen64plus.so.2
src/mk64/player.py          reading the kart out of RDRAM (base 0x0F6990)
src/mk64/ram_hunt.py        how that address was found, by experiment
src/mk64/to_race.py         menus -> start line, once
src/mk64/search.py          rollback segment search (no model needed)
src/plug/input_shm.c        input plugin: controller state in shared memory

src/obs/track.py            centreline, arc length, curvature  (game-agnostic)
src/obs/mk64_adapter.py     N64 -> observation
src/obs/unreal_adapter.py   Unreal -> observation

src/sim/kart.py             vectorised kart; Physics is the knob set
src/sim/calibrate.py        fits the sim to the ROM, reports the residual
src/sim/train_es.py         evolution strategies

src/policy/drive.py         Driver.load(...).drive(env, state)   <- slot-in API
policies/luigi_raceway_v2.*  the working policy + its provenance
states/luigi_raceway_tt_start.st   the start line, as a savestate
tracks/*.json               21 course centrelines from the decomp

scripts/demo_live.py        watch it drive, 60fps
scripts/unreal_server.py    socket server for Unreal
scripts/generalise.py       does it drive tracks it has never seen?
```

**External dependencies:** ROM at `~/Lets Burn Tokens/MK64/Mario Kart 64 (USA).z64`;
decompilation at `~/Lets Burn Tokens/mk64` (course paths, physics constants);
m64p headers vendored from `~/Lets Burn Tokens/simple64`.

---

## 8. If you pick this up next

1. **Finish the generalisation study** and read `runs/generalisation.json`. It
   decides whether the Unreal handoff needs a retrain per map.
2. **Build the confidence layer.** Four policies, one of which fails on real
   hardware — check whether ensemble disagreement predicts *where* it fails. If
   it does, that is the demo's best visual; if it doesn't, say so.
3. **Wire up the Unreal map** with the calibration recording from §3.
4. Do not trust a sim number that has not been checked against the emulator, and
   do not trust an emulator number measured on the wrong bytes (§5).
