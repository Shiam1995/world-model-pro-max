# Mario to Sonic

Solve Mario Kart 64, make the agent show how sure it is, rebuild the courses in
Unreal where physics is a knob, then drive the same policies round a random
track cut through real London. Design and staging in [SPEC.md](SPEC.md).

## Status

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

**5. The attract screen cycles with period 3.**
Any before/after comparison an exact multiple of 3 frames apart shows no change
on a healthy emulator. Two of this project's own tests failed on that before the
code was at fault. Sample consecutive frames, or use a count coprime to 3.

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

## Next

Milestone 3: `ram_hunt` — find the kart's position, speed and heading by
diffing savestates across known motions rather than by disassembly. Expect N64
fixed point, not IEEE floats.
