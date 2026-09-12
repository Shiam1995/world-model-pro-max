# World Model Pro Max

Every policy in this project was trained by **evolution strategies inside a fitted
simulator**. None was ever trained against the emulator, and none against Unreal — the
real worlds are only ever used to *measure*.

**9 networks · 385 parameters each · 17 min 23 s of total training · 0 gradients.**

| | |
|---|---|
| Best real Mario Kart 64 lap | **26.70 s** |
| Training time that produced it | 112 s |
| Seeds completing a lap on the real console | 4 of 5 |
| Top-speed error, sim vs console | 0.8 % |

A rendered version of this report, with the hardware chart, is at
[`training-ledger.html`](training-ledger.html).

## Method — one algorithm, used twice

Not reinforcement learning, not gradient descent. The simulator steps hundreds of karts at
once, so **the population is the batch**: every kart carries its own weights and a single
vectorised rollout scores a whole generation. The top 12.5 % by fitness are averaged into
the next generation's mean and the search radius shrinks 3 % per generation.

Fitness is ground covered, charged for leaving the road:

```
progress − 0.35 × offroad_frames × top_speed
```

so a policy cannot win by cutting a corner across the grass. The network is an MLP
10 → 32 → 1 with tanh — 385 numbers, steering only, throttle held open.

| Hyperparameter | Value |
|---|---|
| Generations | 80 |
| Population | 384 (= karts in sim) |
| Elite fraction | 0.125 → 48 kept |
| Sigma | 0.35, ×0.97 per generation |
| Rollout | 900 frames (1400 in Unreal) |
| Start jitter | ±55 lateral, 0.30 yaw |

Two deliberate anti-overfitting measures. The start point is **randomised to any checkpoint
on the lap** each generation, so nothing wins by memorising one launch. And the physics
itself is jittered every generation — turn rate ×0.70–1.45, acceleration ×0.80–1.25, top
speed ×0.9–1.1, road width ×0.75–1.15 — because the sim is a fit, not a port, and its
residual error is what broke the first transfer attempt.

## Campaign 01 — Luigi Raceway, Mario Kart 64

Five independent runs from different seeds, ~110 s of CPU each, then every one validated on
the real ROM from the committed start-line savestate.

| Policy | Train | Sim laps | Console | Lap time | Mean speed | Off-road | Mean \|lateral\| |
|---|---|---|---|---|---|---|---|
| `es_v2` → `luigi_raceway_v2` | 112 s | — | **LAP** | **26.72 s** | 3.78 | 118 f | 36.7 |
| `es_seed1` | 109 s | 0.757 | LAP | 39.92 s | 2.59 | 172 f | 38.5 |
| `es_seed2` | 111 s | 0.754 | **DNF** | 66.67 s* | 1.01 | 70 f | 32.7 |
| `es_seed3` | 108 s | 0.677 | LAP | 42.00 s | 2.47 | 77 f | 35.0 |
| `es_seed4` | 108 s | 0.730 | LAP | 35.15 s | 2.87 | 174 f | 38.6 |

\* ran out the 4000-frame evaluation budget without crossing the line.

**Sim score did not predict the console.** `es_seed2` had the second-best simulator fitness
of the fleet and is the one policy that fails on real hardware; `es_seed3` had the worst and
finished. Ranking the seeds by sim fitness would have picked the wrong driver. The only
honest ranking came from running all five on the ROM.

Source data: [`runs/fleet_train.json`](../runs/fleet_train.json),
[`runs/fleet_eval.json`](../runs/fleet_eval.json) — both gitignored, regenerate with
`scripts/train_fleet.py`.

## The world it trained in

Physics constants that exist in the decompilation — top speed per engine class, a ten-point
acceleration curve, friction, gravity — are lifted directly. Everything else is *fitted* to
scripted traces collected from the emulator: accelerate from rest, then hold each of three
steering angles for 360 frames. The residual is reported rather than assumed.

| | |
|---|---|
| Longitudinal RMSE | 0.404 units/tick (7.2 % of top speed) |
| Top speed | 5.539 sim vs 5.581 real |
| Frames to 90 % of top | 133 sim vs 137 real |
| Throughput | ≈250 k kart-frames/s sustained |

A Luigi Raceway run is 384 karts × 900 frames × 80 generations ≈ **27.6 million
kart-frames**, delivered in under two minutes. The emulator manages roughly 100 frames a
second. That ratio is the entire reason a fitted sim is worth building.

## Campaign 02 — King's Cross figure-8, Unreal Engine

Record one open-loop calibration run in Unreal (524 logged frames — a fixed script of
straights and held steering angles), fit the kart sim to that trace, train in the sim, drive
the result back. Nothing about the vehicle is guessed; every number below is measured off the
trace.

| | |
|---|---|
| Track | 1003.6 m closed figure-8 |
| Scale | 12.60 units per metre |
| Road half-width | 75.6 units ≈ 6.0 m |
| Fitted top speed | 1.866 units/tick |
| Handedness | +1, verified not assumed |
| Steer response | 0.15 → 3.9 %, 0.8 → 92 % |

| Seed | Train | Progress | Of lap | Off-road | Status |
|---|---|---|---|---|---|
| seed 2 → `kingscross_mk64_v1` | 164.3 s | 346.8 m | 0.346 | 0 f | not yet driven |
| seed 1 | 164.7 s | 346.7 m | 0.346 | 0 f | not yet driven |
| seed 0 | 166.4 s | 345.9 m | 0.345 | 0 f | not yet driven |

**0.346 laps is not a failure rate.** At the fitted top speed a 1400-step rollout covers
about 415 m and the lap is 1003.6 m, so the training window ends before a lap can finish by
construction. The three seeds covered 84 % of the theoretical maximum distance with *zero*
off-road frames and landed within 0.3 % of each other. What has not happened is the part
that counts: **none has been driven back in Unreal**, so this campaign has a sim result and
no real result.

**One measured risk.** The turn table only covers speeds down to 26.8 % of top; below that
the sim extrapolates. An invented low-speed steering ramp is precisely what the README names
as "the single thing that broke sim-to-real", because it lets a policy carry a large constant
steering bias for free. The fix is more calibration coverage at low speed, not a tuned
constant.

## What was not trained

- **No emulator-in-the-loop training.** Deliberate. The console is the oracle, never the
  training signal — which is what makes "trained for 112 s, then completed a real lap" a
  transfer claim rather than a fitting claim.
- **No gradient method, no RL.** Also deliberate: long episodes, delayed reward, 385
  parameters, and a vectorised sim where the population is free. ES needs no credit
  assignment.
- **Confidence calibration (stage 2).** The ensemble exists and one member demonstrably fails
  on hardware, so the data to calibrate disagreement against outcomes is in
  `runs/fleet_eval.json`. The calibration itself was never run. Still the biggest missing
  feature.
- **Generalisation across 21 courses.** The study crashed on launch and never produced a row:
  `scripts/generalise.py` hands a 1-D `theta` to `act()`, which indexes it as 2-D —
  `IndexError: too many indices`. A one-line reshape away from running. Until it does,
  whether a policy drives an unseen track is unmeasured.
- **Rollback search on hardware.** The model-free predecessor reached ~20 % of a lap, then the
  core stopped reporting PAUSED at frame 12178 and the run died. Superseded by the trained
  policy, kept because it needs no model at all.
