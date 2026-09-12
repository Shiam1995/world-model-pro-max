# Mario to Sonic

Solve Mario Kart 64, make the agent show how sure it is, rebuild the courses in
Unreal where physics is a knob, then drive the same policies round a random
track cut through real London. Design and staging in [SPEC.md](SPEC.md).

## Status

**Stage 1, milestone 4 — the map and the observation vector — done.** The
course centreline is extracted from the decomp (`tracks/luigi_raceway.json`, 631
points), fitted into a `Track` with arc length and curvature, and turned into
the ten-number vector every world will share.

```
make test-obs
[1] observation v1: 10 finite numbers
[2] straight: |lateral| <= 1.1, |heading err| <= 0.2 deg, progress monotonic
[3] corner preview leads: curv+2400 1.34 vs curv+300 0.51
[4] no absolute position, lap or course identity in the vector
```

Driving straight off the line, the vector reads the world correctly and then
watches the kart fail, which is the useful part:

```
 frame | speed | lateral | head_err | curv+300 curv+1200
   182 |  5.51 |    -0.2 |     0.1d |    -0.00     -0.51   <- corner seen ahead
   262 |  5.56 |    -1.1 |    -0.2d |    -0.51     -1.27   <- corner arrives
   342 |  5.14 |    70.9 |   -33.0d |    -1.27     -1.34   <- not turning
   502 |  0.80 |    80.7 |   -88.3d |    -1.34     -1.32   <- into the wall
```

**Stage 1, milestone 3 — the kart found — done.** `src/mk64/player.py` reads
position, velocity, speed, heading and lap straight out of RDRAM. The address
(`0x0F6990`) was *measured*, not looked up — see below.

```
make test-player
[1] parked at [-139.0, -44.7, -180.0], lap 0, speed 0
[2] cruising at 5.48/tick, consistent (speed == |velocity| == |pos-oldPos|)
[3] 2.00 physics ticks per emulated frame, as calibrated
[4] left vs right diverge by 100 units after 120 frames
```

**Stage 1, milestone 2 — the start line — done.** The menus are walked once by
`src/mk64/to_race.py` and the result is committed as
`states/luigi_raceway_tt_start.st`. Every episode now begins from that file:
boot, `load_state_file(...)`, drive. No menu is ever touched again.

```python
with MK64Env() as env:
    env.step(frames=60)
    env.load_state_file("states/luigi_raceway_tt_start.st")
    env.step(frames=150, accel=True)     # Mario drives down Luigi Raceway
```

**Stage 1, milestone 1 — the harness — done.** Mario Kart 64 runs in-process
under Python with exact frame stepping, analog control, readable RDRAM and
working rollback.

```
make                      # build the input plugin
make test                 # prove the harness (asserts, not prints)
```

```
[1] booted headless in 0.4s, frame=0
[2] 120 frames in 1.68s = 71 fps headless (frame 0->119, polls +245)
[3] RDRAM live across 7 frames: 3 distinct states
[4] rollback exact — save_state() is a usable env.clone()
[5] deterministic replay
ALL CHECKS PASSED
```

```python
from src.mk64.env import MK64Env

with MK64Env() as env:
    env.step(frames=300)                      # skip the logo
    s = env.save_state()                      # == env.clone()
    env.step(frames=60, accel=True, steer=-1.0)
    env.load_state(s)                         # exact, frame for frame
```

## Why in-process

`libmupen64plus.so.2` is loaded into the Python process with ctypes rather than
spawned as a subprocess, so `save_state()`/`load_state()` are function calls on
memory we can see. That is the whole basis of the plan: the savestate is the
world model, and segment search needs `env.clone()` to be cheap. A subprocess
reading pixels off a screen could never do it.

Observations come from RDRAM, never from pixels.

## What bit, and what the fix was

Five things, none of them documented where you would look. Written down because
each one presents as "the AI learned nothing" rather than as an error.

**1. Plugin attach order is load-bearing, and the core only whispers.**
It must be **GFX → AUDIO → INPUT → RSP**. `plugin_connect_rsp()` hands the RSP
whatever gfx function pointers are current, so attaching RSP first silently
binds it to `dummy_gfx`. MK64 then boots, renders framebuffer `0x27f` forever,
fires exactly one VI and never reads the controller again. The only hint is one
`M64MSG_WARNING`: *"Front-end bug: plugins are attached in wrong order."*
Diagnosed by counting `UpdateScreen` calls before the framebuffer origin
changes: stock moves on at call #18, ours never did after 464.

**2. There is no true headless, because the RDP has to be real.**
With no video plugin the RSP gets `dummy_gfx` and MK64 wedges exactly as in (1).
So "headless" here means a window we ignore, not a plugin-less core. Harmless —
observations are RAM-based anyway, and stage 2 wants video regardless.

**3. `GbCameraVideoCaptureBackend1` defaults to `"opencv"` and hangs startup.**
The core opens an OpenCV capture device for Game Boy Camera support before the
CPU ever runs, and blocks on hardware that is not there. This hangs the *stock*
`mupen64plus` binary too. Set it to `""`.

**4. MK64's framebuffers live inside RDRAM, so never hash all of it.**
Origins sit around `0x31fa00`/`0x345200`/`0x36aa00`, and glide64mk2 renders
through an FBO without writing those pages back byte-identically. Hashing all
8 MiB reports a perfectly deterministic emulator as non-deterministic. Compare
`0x000000–0x300000` (`env.game_state_bytes()`).

**5. ~~The attract screen cycles with period 3.~~ — WRONG, and worth keeping
visible.** Two tests really did fail this way, but the cause was trap 8 below:
the "RAM" being hashed was the VI register block, which cycles through the three
framebuffer origins. Reading actual RDRAM gives 7 distinct states across 7
frames. A tidy explanation that fits the evidence is not the same as the cause.

Also: a savestate has to carry the **controller** too. The pad lives outside the
emulator, so restoring RAM alone lets the settling frame replay a different
input, and the rollback is no longer exact.

## Layout

```
SPEC.md                       the four stages, and the one observation vector
Makefile                      builds the input plugin
vendor/m64p/                  m64p ABI headers (from simple64's core source)
src/plug/input_shm.c          input plugin: controller state in shared memory
src/mk64/core.py              ctypes binding to libmupen64plus.so.2
src/mk64/pad.py               N64 controller bit packing + the shm writer
src/mk64/env.py               MK64Env: step / save_state / load_state / ram
src/mk64/selftest.py          the five checks above
```

The input plugin exists because the alternative — injecting fake SDL key events
— can only ever steer full-lock. MK64 steering is an 8-bit signed axis, and a
policy that cannot hold a partial steering angle cannot hold a racing line.

## Two more that cost time

**6. MK64 puts an "OK ?" confirm after *every* selection.** It is **five** A
presses from PLAYER SELECT to the track, not four. Four leaves you on MAP SELECT
staring at a screen that looks like it is ignoring input — and it is not; it is
waiting for the confirm you never sent. Cross-check any "the game is frozen"
theory against `pad.polls`: if it keeps rising, the game is alive and reading
you, and the bug is yours.

**7. `M64CORE_STATE_SAVECOMPLETE` fires before the bytes reach disk.** The
savestate is written on the core's worker thread, so right after the signal the
file is a truncated gzip — 16 KiB standing in for a 16 MiB state. It exists, it
has the right magic number, and it is useless. Wait until the file actually
decompresses (`_await_complete_state_file`), not until the core says it is done.

Related: **the frame callback is not a completion signal.** It runs inside
`new_frame()`, *before* the core's `if (l_FrameAdvance) { g_rom_pause = 1; ... }`
bookkeeping. Request the next frame in that window and the emulator thread eats
the flag, pauses anyway, and never resumes — an intermittent hang that reads as
a slow frame. Wait for `M64EMU_PAUSED` instead.

## Finding the kart (milestone 3)

`src/mk64/ram_hunt.py` runs four trajectories out of one savestate — `still`,
`fwd`, `left`, `right` — and looks for memory that moves under throttle, sits
exactly still without it, and splits opposite ways under steering. Rollback is
what makes this a laboratory rather than a fishing trip: every trajectory starts
from the identical instant, so any difference was caused by the input.

The struct *layout* came from the mk64 decomp (`pos` 0x14, `oldPos` 0x20,
`velocity` 0x34, `speed` 0x94, size 0xDD8). The *address* had to be measured —
which is the half that will still work on Sonic Riders, where there is no decomp
to read.

The winner scored 16/16 and proves itself:

```
still  pos [-139.0, -44.7, -180.0]   speed 0.00   lap 0
fwd    pos [-139.0, -44.7, -785.3]   oldPos [-139.0, -44.7, -780.5]
       vel [0.0, 0.0, -4.82]         speed 4.82
```

`speed` == `|velocity|` == `|pos - oldPos|` == 4.82, three fields stored
separately agreeing to the last digit. Nothing lands on that by coincidence, so
`player.check()` keeps asserting it at runtime: if the struct ever moves, it
fails loudly instead of feeding the policy noise.

**8. `M64P_DBG_PTR_RDRAM` is 1, not 4.** Four is `M64P_DBG_PTR_VI_REG`, a small
register block — so `rdram()` returned a pointer to the VI registers and every
read ran megabytes off the end of it. It did not crash reliably, because whether
it segfaults depends on heap layout; under gdb it survived and without it the
process dumped core. It also quietly invalidated earlier results: the first
"rollback exact" was measured against the wrong bytes, and trap 5 was pure
artefact. **A test that passes against the wrong memory is worse than one that
fails.**

**9. RDRAM is stored in HOST word order — little-endian on x86.** Not the N64's
big-endian layout. Scanning as `>f4` finds nothing real; the giveaway is that a
big-endian scan turns up **zero** words incrementing by 1 per frame, and every
running game has a frame counter somewhere. Little-endian found 13 immediately
(there is one at `0x0E9E24`).

**10. `RandomizeInterrupt` defaults to on.** The core jitters PI/SI interrupt
timing deliberately. The same actions from the same savestate then diverge by
~100 bytes — small enough to look like a subtle bug of your own, fatal to
segment search. Turn it off and replay is bit-exact.

**11. MK64 runs two physics ticks per rendered frame.** `speed` and
`pos - oldPos` are per *tick*, so the kart covers `2 x speed` per emulated
frame — measured at 2.00 exactly. A silent factor of two in every distance,
reward and lookahead otherwise.

## The map, and a plan that did not survive (milestone 4)

SPEC said: drive a lap, fit a centreline from it, drop checkpoints. The driving
part failed, and the failure is worth keeping.

`src/mk64/ghost.py` searches out a lap with no map at all — branch the savestate
across five steering choices, keep whichever covers the most ground. It works
for about thirty segments and then wedges the kart against the outside wall of
turn one at speed 0.12. Greedy "cover the most ground in the next half second"
cannot tell a corner from a circle, and once stuck, every branch scores zero, so
it has no gradient back out. MK64's own `nearestPathPointId` would have supplied
real progress, but it stays 0 for the player in a time trial.

So the map comes from the road instead: MK64 ships the path its CPU racers
follow, and `scripts/extract_course_path.py` reads it out of the decomp
(`d_course_luigi_raceway_track_path`, 631 points at 20-unit spacing). The kart's
start position projects **0.5 units** from that centreline, which is the
confirmation that the extracted path and the live game agree.

That is the same bargain stages 3 and 4 strike anyway — Unreal takes geometry
from the decomp, London from OSM. Driving blind is only needed where no map
exists, and `ghost.py` stays in the tree for exactly that case.

Watch for: the path array ends in a `-32768` sentinel row. Leaving it in puts a
point 32,000 units away and quietly wrecks every arc length and projection.

## The observation vector

`src/obs/track.py` is game-agnostic — a smoothed centreline, arc length,
curvature — and `src/obs/mk64_adapter.py` is the only file that knows both Mario
Kart and the vector. Unreal and London get an adapter of the same shape and
nothing else changes.

```
speed, lateral_offset, heading_sin, heading_cos,
curv_300, curv_600, curv_1200, curv_2400, grade, progress_delta
```

Deliberately contains nothing that identifies *which* track it is: no absolute
position, no lap, no course id. A policy that cannot tell Luigi Raceway from a
London roundabout is one that can be moved between them — and `make test-obs`
asserts that.

Ten numbers, not the twenty SPEC sketched. The rangefinders and track width are
**missing rather than faked**: both need track *edges*, and Luigi Raceway ships
only a centreline. They arrive with real geometry in stage 3, and `OBS_VERSION`
goes up when they do.

## Next

Milestone 5: behaviour-clone a lap so training never starts from random, then
improve it by savestate segment search — now with a real progress signal
(`track.project`) to search against, which is exactly what the greedy driver
lacked.

---

## Slot-in policy

A policy trained **entirely in simulation**, in 112 seconds, that completes a lap
of Luigi Raceway on the real Nintendo 64. It never saw the emulator during
training.

```python
from src.mk64.env import MK64Env
from src.policy.drive import Driver

with MK64Env() as env:
    d = Driver.load("policies/luigi_raceway_v2.npy")
    print(d.drive(env, "states/luigi_raceway_tt_start.st"))
# {'completed_lap': True, 'seconds': 26.72, 'mean_speed': 3.78, ...}
```

| policy | real lap | time | mean speed | off-road |
|---|---|---|---|---|
| **luigi_raceway_v2** | yes | **26.72 s** | 3.78 | 118 / 1603 |
| es_seed4 | yes | 35.15 s | 2.87 | 174 / 2109 |
| es_seed1 | yes | 39.92 s | 2.59 | 172 / 2395 |
| es_seed3 | yes | 42.00 s | 2.47 | 77 / 2520 |
| es_seed2 | no | — | 1.01 | — |

Four of five seeds transfer, so it is reproducible rather than a lucky run. The
same lap repeats bit-identically three times out of three.

It is 385 numbers and an MLP with ten inputs. Nothing in it knows what Mario
Kart is — hand it another world that emits the same ten numbers and it drives
that one instead. That is the whole point of the observation vector.

### What finally made transfer work

Three real differences between the sim and the console, each found by
measurement rather than argument:

1. **The turn was mirrored.** Replaying one fixed action script in both worlds
   showed `+0.8` steer moving the real kart toward `+x` and the sim kart toward
   `-x`. They agreed to 31 units while driving straight and were 678 apart after
   two turns. *This was the one that mattered.*
2. **Steering is non-linear with a deadzone.** Stick 0.15 gives exactly zero yaw
   rate and 0.30 gives 5% of full lock. A policy trained on a linear model
   steers with gentle corrections that do nothing on the real kart.
3. **Heading now comes from the `velocity` vector**, not the yaw angle — a real
   vector needs no sign or axis convention guessed about it.

Ruled out along the way, each by measurement: steering sign at the policy output
(A/B on the emulator), the yaw→heading trig convention, and parameter error in
general (domain randomisation did not help, which is what said the gap was
systematic).
