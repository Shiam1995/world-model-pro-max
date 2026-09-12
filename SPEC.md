# Mario to Sonic

Solve Mario Kart 64, make the agent show how sure it is, rebuild the courses
where physics is a knob, then drive the same policies round a random track cut
through real London.

Started 2026-09-12.

---

## The spine: one observation vector, four worlds

Everything in this project hangs off a single decision.

Every world — the N64 emulator, the Unreal rebuild, the London track — exposes
the **same ~20-number track-relative observation**, and every policy is written
against that vector, never against a game:

```
speed, lateral_accel, heading_error, lateral_offset,
curvature at +10/20/40/80 m, width at +10/20/40 m,
7 rangefinders, on_track, airborne, progress_delta
```

No pixels. No RAM addresses above the adapter layer. No Mario Kart concepts.

This is the whole reason the four stages compose instead of being four projects:
a policy trained on Royal Raceway can be dropped into a London roundabout
because neither one is in its input. Only the **adapter** differs per world.

```
           ┌──────────────┐
           │   policy     │  ensemble of N heads -> action + confidence
           └──────┬───────┘
                  │ 20 floats in, 3 floats out (steer, accel, brake/hop)
   ┌──────────────┼──────────────┬──────────────────┐
   │              │              │                  │
mk64 adapter  unreal adapter  london adapter    (sonic riders, later)
 RAM reads     gameplay tick   same as unreal
 savestates    engine physics   OSM centreline
```

## Stage 1 — Solve Mario Kart 64

No decompiling *for the agent*: the decomp is used for geometry in stage 3, but
the agent learns from the running ROM, the way the other racing project does.
The emulator savestate IS the world model: `save_state()` is `env.clone()`.

1. **Harness.** ctypes against `libmupen64plus.so.2` + a small custom input
   plugin (shared-memory controller state). `--gfx dummy --audio dummy` for
   headless speed; real video plugin on demand to watch.
2. **Find the car by savestate diffing, not disassembly.** `ram_hunt` —
   drive, snapshot RAM, correlate candidate words against known motion.
   Expect N64 fixed point, not IEEE floats.
3. **Ghost -> Map.** Drive one lap by hand, log positions, fit a centreline,
   drop virtual checkpoints every 10 m. Dense reward, zero game knowledge.
   (The Linesight recipe.)
4. **Clone yourself.** Behaviour-clone the human lap so training never starts
   from random flailing.
5. **Improve by savestate search.** Segment-level expert iteration: branch from
   a state, search actions, keep the best, distil back into the policy.

**Done when:** the policy beats the 150cc staff ghost on 3 courses it trained on
and finishes 3 it never saw.

## Stage 2 — Confidence while driving

Confidence must be *earned*, not decorated. Two rules:

- It is **ensemble disagreement**, not `abs(steering)`. N heads trained on
  different data orders; confidence = inverse spread of their proposed actions.
  A single net's softmax is not a confidence signal and will not be used.
- It is **measured against being right**. A confidence number nobody checked is
  a dial painted on a box. Every run logs (confidence, outcome) and plots
  calibration: when it says 90%, it should hold the line 90% of the time.

Rendered as a live overlay on the race: a confidence ribbon along the racing
line ahead, and per-head phantom steering inputs fanned out around the wheel.
Where the heads agree the ribbon is tight; entering a corner it never mastered,
it frays. **This is the money shot of the project.**

## Stage 3 — The courses in Unreal, physics on a knob

1. Extract course geometry from the mk64 decomp: `make model_extract` /
   fast64 -> Blender -> FBX -> Unreal (the Characters-project pipeline).
   Caution: repo fast64 targets Blender 3.6; this box has 5.2.1.
2. Rebuild drivable track + collision in Unreal, expose the same 20-number
   observation from a gameplay tick.
3. **Physics becomes a parameter set**, not code: grip, mass, drift authority,
   hop, gravity, air control, ground clearance. One struct, swappable presets.
4. Dial it from kart toward hoverboard — that is where "Mario to Sonic" lands.
   Re-measure the stage-1 policy at each setting: where does it break, and does
   the stage-2 confidence see the break coming before the lap time does?

**Done when:** the same policy file drives the Unreal rebuild of a course it
learned on the N64, and a physics slider degrades it measurably and visibly.

## Stage 4 — London

1. One OSM extract of Greater London (Geofabrik, a single no-key download —
   the deliberate egress exception, like Crossref in Synapse).
2. Build a road graph, then **generate a random circuit**: a closed loop of
   real streets, sane length, no impossible turns, start/finish that joins up.
3. Same adapter as Unreal, so the policies need no retraining to try it.
4. Race the karts round it. This is the generalisation test — and the honest
   one, because nothing about London is in the training set.

**Done when:** N karts complete a randomly generated London circuit, and the
confidence trace over the lap is legible: low where the geometry is unlike
anything Nintendo drew.

---

## Rules

- **Local-first.** Local models, vendored deps, no external APIs beyond the
  single OSM extract.
- **The adapter is the only game-specific code.** If a policy or a trainer
  mentions Mario Kart, it is in the wrong file.
- **A policy file names the world it was trained on** and records which
  observation version it expects. Refuse to load a mismatch.
- **Nothing is "solved" on training tracks.** Every number quoted is from
  held-out courses.
- **Show the changes.** Comment out superseded approaches rather than deleting
  them; keep the finding visible.
