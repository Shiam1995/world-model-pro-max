# What to record in Unreal before plugging the agent in

Two files. Takes about five minutes of driving.

## 1. track.json — once

```json
{"type": "track",
 "centreline": [[x,y,z], ...],   world-space points along the middle of the road,
                                 in driving order, last point adjacent to the first
 "half_width": 350.0,            half the drivable width, same units
 "up_axis": "z",                 "z" for stock Unreal
 "units_per_metre": 100.0}       100 for Unreal (1 uu = 1 cm)
```

Spacing of 2-5 m between centreline points is ideal. Denser is fine, it gets
resampled.

## 2. calibration.jsonl — one line per frame

Drive the car with this **fixed script**, recording every frame. Do not steer by
hand and do not use any AI — the whole point is that the inputs are known:

| frames | throttle | steer |
|---|---|---|
| 0-89    | 1.0 | 0.0  |
| 90-149  | 1.0 | +0.8 |
| 150-209 | 1.0 | 0.0  |
| 210-269 | 1.0 | -0.8 |
| 270-359 | 1.0 | 0.0  |

```json
{"frame": 0, "steer": 0.0, "throttle": 1.0, "pos": [x,y,z], "vel": [x,y,z]}
```

That one recording answers everything that matters:

- **Handedness** — whether `+steer` turns the car the way the model thinks. This
  is the bug that cost this project a day on the N64 side, and an open-loop
  replay finds it in one run.
- **Acceleration curve and top speed** — measured, not guessed.
- **Steering response** — how much yaw rate each stick position actually buys,
  including any deadzone.

With those two files a policy native to *your* track and *your* vehicle trains
in about two minutes.
