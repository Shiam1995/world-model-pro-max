# Stage 4 (London) — data & pipeline research

Researched 2026-09-12. Three parallel threads: UK open elevation, road geometry
and width, OSM→drivable-road pipelines. Figures below were **measured**, not
quoted from spec sheets — OSM tag counts via Overpass over the bbox, elevation
statistics from the actual EA raster pulled over the same bbox.

Bbox throughout: `51.498–51.509 N, -0.144 to -0.120 E`
(= BNG E 528931–530566, N 179306–180572; 1.63 × 1.27 km).

---

## 0. The finding that reframes everything

The policy is **pixel-free**. It sees only:

```
speed, lateral_accel, heading_error, lateral_offset,
curvature at +10/20/40/80 m, width at +10/20/40 m,
7 rangefinders, on_track, airborne, progress_delta
```

So "high-fidelity London" does not mean a good-looking London. Visual fidelity —
textures, buildings, road markings, meshes — is worth **zero**. Only four
channels matter: centreline curvature, carriageway width, the drivable-surface
boundary, and elevation *only* via `lateral_accel` and `airborne`.

**Consequence: no mesh, no physics engine.** Build an analytic track model and
query it. See §4.

---

## 1. Width — the weak link, and it is weaker than expected

Measured over the bbox: of **561 driveable OSM ways**, **6 carry `width` (1.07 %)**.
`est_width`: **0**. The six values are `7, 7, 7, 7, 4.3, "10 m"` — four identical
copy-pasted defaults, and one unit-suffixed string that breaks naive float parsing.

**Treat OSM `width` as not a data source. Do not write a code path that expects it.**

What *is* populated: `maxspeed` 98.9 % (but useless — 544/551 ways are 20 mph),
`surface` 90.6 %, `lanes` 54.7 % of ways / 46.6 % by length, `oneway` 66.1 %.

### No open road polygons exist
Checked and negative: OS Open Roads, OpenMap Local, Zoomstack, VectorMap District
are all `GM_Curve` — cartographic lines. OSM `area:highway` has 71 ways in the
bbox but only **6 are carriageway**. TfL's 40 open datasets carry no widths or
lane counts. The products that *do* have true width — OS NGD `roadwidth_average`,
OS MasterMap Topography road polygons — are all behind an account wall.

### Three options
1. **Infer**: `lanes × lane width` (Manual for Streets: 2.75 / 4.10 / 4.80 / 5.50 m
   carriageway; central-London lane 3.0–3.25 m, bus lane 4.0–4.5 m; DMRB lane 3.65 m).
   Accuracy **±1.0–1.5 m on a ~7 m carriageway (±15–20 %)**.
   Trap: OSM wiki says `lanes` is a **minimum, not an exact count**.
2. **Constant default** (SUMO 3.2 m, CARLA/osm2xodr 4.0 m) — this makes
   `width at +10/20/40 m` constant and **silently deletes three of the twenty observations**.
3. **Trace it** from EA Vertical Aerial Photography (OGL, no login, 10–50 cm GSD).
   At 12.5–25 cm, London kerb lines are unambiguous. One afternoon per circuit.
   This is what TUM did for `TUMFTM/racetrack-database`.

**Systematic bias on top:** kerbside parking is mapped on **3 of 551 ways**, so
nominal kerb-to-kerb width overstates drivable surface by ~2 m on Westminster streets.

**Best structured helper:** `a-b-street/osm2streets` (Rust, Apache-2.0, active) —
emits roads as thickened line-strings with per-lane type/direction/width **and
intersections as polygons**. That is a drivable boundary *and* a junction surface,
offline, without meshing.

---

## 2. Elevation — essentially flat, and the raw data will lie to you

Measured over 2,069,910 cells of 1 m DTM:

| | m OD |
|---|---|
| Land min / max | 1.0 / 21.8 |
| **Total relief** | **20.8 m** |
| Thames water surface | 3.11 (N of bridge) / 3.25 (S) |

Gradient at 100 m smoothing: p50 **1.44 %**, p90 3.86 %, max 10.03 %.
Mean macro gradient NW→SE **0.90 %**. The whole bbox is one broad ramp from
Green Park (~21 m) down to the Thames (~3 m). No local structure.

Real streets: Whitehall ±1.3 %, The Mall −2.6 %, Birdcage Walk −3.8 %,
Constitution Hill +3.6 %, Haymarket/Lower Regent St steepest at ~3–4 % sustained.
A 4 % grade is worth 0.39 m/s² — under 5 % of braking authority.

### THE TRAP: raw 1 m DTM produces 17.7 g of fake vertical acceleration

Measured on 300 m of dead-straight Whitehall at 20 m/s:

| Along-track smoothing | min crest R | peak vertical accel | airborne fires? |
|---|---|---|---|
| raw 1 m | 2.3 m | **17.7 g** | YES |
| 11 m | 11.0 m | 3.7 g | YES |
| 31 m | 27.3 m | 1.49 g | YES |
| **41 m** | 46.2 m | 0.88 g | no |
| 71 m | 312.8 m | 0.13 g | no |

**Smooth to ≥70 m along-track before sampling as a driving surface**, or the car
flies off a flat street on measurement noise. Once smoothed to 70 m, only the
19 m macro ramp survives — every fine detail you paid for must be destroyed
before the data is usable.

**Take the 1 m LiDAR anyway — but because it is 16 MB and one `curl`, not because
the resolution buys anything.** Coarse alternatives lose on *accuracy*, not
resolution: OS Terrain 50's 4 m RMSE against 19 m of relief is SNR < 1 and would
invent several percent of fake gradient on flat streets. EA at ±0.15 m gives SNR ≈ 7.

### `airborne` will never fire
Needs `v²/R > g` → at 20 m/s, a crest radius under 40.8 m. Across 192 open-ground
runs of ≥150 m, the **median tightest crest radius is 190–220 m**. Nothing on a
carriageway comes close. Either place synthetic crests deliberately, or accept
that this observation is permanently zero.

### THE OTHER TRAP: Westminster Bridge puts the car underwater
The EA DTM is **bare earth** — bridges are not bare earth. Sampling east along
N 179672:

| E | DTM | DSM | |
|---|---|---|---|
| 530330 | 8.55 | 8.56 | approach |
| **530335** | **3.95** | 8.72 | **5.4 m cliff in one pixel** |
| 530480 | **3.15** | 10.5 | mid-span = Thames water surface, flat-filled |

True deck ~9.3 m OD → **DTM error on the deck: 6.1 m**. The fill constants even
differ north vs south of the bridge (3.11 vs 3.25 m — different survey epochs and
tide states), leaving a 0.14 m step mid-river.

**The DSM does not rescue you**: its bridge values are parapet and lamp-post tops,
and elsewhere it is catastrophic — Charing Cross viaduct over Villiers Street reads
DTM 4.31 m (correct ground) vs **DSM 49.96 m (station roof)**.

**Everything else in the bbox is correct**: Admiralty Arch (road through a
building) 5.88 m, Hungerford footbridge 4.32 m, Duke of York Steps 11.01 m,
Embankment river wall 3.11 m. There are **no road tunnels or underpasses in the
bbox** (Hyde Park Corner and Strand/Kingsway are outside it).

**Mitigation: hardcode Westminster Bridge as a flat deck at ~9.3 m OD with ~3 %
approach ramps.** One polygon, ~20 minutes. No resolution fixes this — it is a
data-model decision, not a sampling limit.

**Also despike**: 0.18 % of the bbox carries impossible sub-zero values (DTM to
−10.83 m, DSM to −40.19 m) along the Embankment near Hungerford Bridge.

### Camber: synthesise it — but not for the reason first assumed
**Correction worth recording.** ±0.15 m is the EA's *absolute survey RMSE*, not the
local noise floor. Measured relative roughness by plane-fitting 30 × 30 m patches:
**27 mm σ on Whitehall carriageway** (73 mm The Mall, 169 mm Horse Guards gravel).
So a 17.5 cm crossfall is ~6σ above local noise, and the Whitehall crown *was*
recovered by stacking 280 transects (−304 mm at the west building line, **+125 mm
plateau at 16–17 m offset**, −68 mm at 37 m).

Synthesise anyway, for the real reasons:
- the 1 m cell straddles the crown, so recovered crossfall is ~1 % against a 2.5 %
  design value — **understates by 2×**;
- it yields one number per street, not per-location geometry;
- the effect is **0.245 m/s² against ~4.6 m/s² of cornering (~5 %)** and
  **sign-flips at the crown**, so lane-averaged it is ≈ 0.

Use DMRB CD 109: `S% = V²/(2.828·R)`; `V²/R < 5` → 2.5 % camber from the crown;
max **5 % urban**. Smooth the superelevation ramp over a runoff length.

**Kerbs are unrecoverable at any open resolution.** The largest single 1 m step
anywhere across Whitehall is **33 mm** against a real 100–125 mm BS 7263 upstand.
You would need ≤10 cm GSD and ≤20 mm σ_z — mobile laser scanning, which does not
exist openly for London.

### WCS recipe (verified: HTTP 200, 16 MB, 4.0 s, plain curl, no key, no cookie)
```
https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs
  ?service=WCS&version=2.0.1&request=GetCoverage
  &coverageId=13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m
  &format=image/tiff
  &subset=E(528800,530800)&subset=N(179000,181000)
```
Tight fit: `subset=E(528931,530566)&subset=N(179306,180572)` → 1635 × 1266.
DSM (bridge-deck reference): path `…-digital-surface-model-last-return-dsm-1m/wcs`,
coverageId `9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m`.

- **CRS EPSG:27700** (BNG), heights ODN via OSTN'15. Axis labels `E N`, metres.
- Returns uncompressed big-endian single-band **float32 TIFF**, row 0 = north,
  NoData `-3.4028235e38`.
- **No GDAL needed** — PIL opens it as mode `F`, numpy takes it directly.
- Licence **OGL v3**: `© Environment Agency copyright and/or database right 2022.
  All rights reserved.` Accuracy ±0.15 m RMSE. Surveys 2017-12→2018-01 and 2020-12.

**Gotchas**
1. **Do not sanity-check by byte count.** Two requests of *different* dimensions
   both returned exactly 12,583,427 bytes (fixed buffer + padding). Check the
   decoded array shape.
2. Returns the native grid — two requests with different origins overlapped at
   100.00 % identical pixels, max diff 0.000000 m. No server-side resampling.
3. The old ArcGIS route (`gp/DataDownload/GPServer`) is **dead** — 400 Invalid URL.
   Any script found online using it will fail.
4. The portal's "Create an account" is a red herring — that is the interactive
   order basket. Its backend `/api/survey/download` returns **403** anonymously.
   The WCS takes no credentials (`Fees: NONE`, `AccessConstraints: NONE`).
5. Scales fine — a 10 × 10 km request (~400 MB) also returns 200.

Storage: bbox 8 MB, 2 km tile 16 MB, central London 400 MB, all Greater London 8 GB.

---

## 3. Sources — verdicts

| Source | Verdict |
|---|---|
| **OSM via Geofabrik** `greater-london-latest.osm.pbf` (128.7 MB, ODbL, EPSG:4326) | **Primary.** Best centreline, curvature, topology. 57 turn-restriction relations in bbox |
| **EA LIDAR Composite DTM 1 m** (WCS, OGL v3, ±0.15 m) | **Use.** See §2 |
| EA Composite DSM 1 m | Bridge-deck reference and building heights (DSM−DTM) only |
| **EA Vertical Aerial Photography** (10–50 cm, OGL, no login) | The only no-key route to *true* width — by manual tracing |
| `a-b-street/osm2streets` (Apache-2.0) | Lane widths + **intersection polygons** |
| OS Open Roads | **Reject.** Centreline only, generalised to 1:25 000 (~10 m displacement), "Collapsed Dual Carriageway" merges both carriageways into one line |
| OS Open Zoomstack / OpenMap Local / VectorMap District | Reject — lines, no width; buildings have no height |
| OS NGD (`roadwidth_average`) / MasterMap Topography | **The one true-width source. Account-gated → disqualified by the local-first rule** |
| OS Terrain 50 | Reject — 4 m RMSE, SNR < 1, and GB-only 154 MB zip for ~202 KB of tiles |
| OS Terrain 5 | Disqualified — absent from the open catalogue, paid |
| Copernicus GLO-30 | Reject — DSM, 54 % of cells >3 m off; also ESA/Airbus licence, not OGL |
| SRTM / NASADEM / ASTER | Disqualified — 401 → Earthdata login. (Key-free mirror exists: `opentopography.s3.sdsc.edu`) |
| CGIAR-CSI SRTM 90 m | Licence fails — no redistribution without written permission |
| TfL GIS Open Data Hub (40 datasets) | No widths, no lane counts, no carriageway polygons |
| London Datastore (1,296 packages scanned) | **Zero elevation datasets** |
| Overture Maps | Keyless S3 Parquet, but width is conflated *from OSM* — same sparsity, no new information |

---

## 4. Runtime: analytic model, not mesh + physics engine

**esmini's standalone RoadManager is the observation vector, already implemented.**
`github.com/esmini/esmini`, **MPL-2.0**, v3.8.1 (2026-09-08), weekly cadence.
Headless build: `-D USE_OSG=False -D USE_SUMO=False -D USE_OSI=False`.
CMake downloads prebuilt third-party packages at *configure* time; runtime is then
fully offline. No PyPI package — write ~60 lines of ctypes against the flat,
handle-based C API in `esminiRMLib.hpp`.

```c
RM_GetProbeInfo(handle, lookahead_distance, RM_RoadProbeInfo*, lookAheadMode, inRoadDrivingDirection)
// -> RM_RoadLaneInfo { heading, pitch, roll, width, curvature, laneOffset, s, t, lane_type, ... }
```

| Observation | Call |
|---|---|
| curvature at +10/20/40/80 m | `RM_GetProbeInfo` once per lookahead |
| width at +10/20/40 m | same (note: `.width` is *lane* width — sum drivable lanes for road width) |
| lateral_offset, heading_error | `RM_SetWorldXYHPosition` + `RM_GetPositionData` → `laneOffset`, `hRelative` |
| on_track | `RM_GetInLaneType` |
| progress_delta | `RM_SubtractAFromB` |
| junction ambiguity | `RM_GetNumberOfRoadsOverlapping` / `RM_GetOverlappingRoadId` |

**Curvature is analytic, not finite-differenced** — `EvaluateCurvatureDS()` is
implemented per geometry primitive (Line/Arc/Spiral/Poly3/ParamPoly3) in
`RoadManager.cpp`.

**Prior art converges independently.** TORCS/Speed Dreams tracks are ordered
segment lists (straight+length or turn+arc+radii, per-segment width/banking/grade)
— no mesh in the source of truth. Assetto Corsa's `ai/fast_lane.ai` stores
`Radius, SideLeft, SideRight, Camber, Direction, Normal, Grade` per point — AC
**re-derives this observation vector back out of its own mesh in order to drive**.
F1TENTH's raceline CSV is `[s, x, y, psi, kappa, vx, ax]` — the vector again.
AC modders' rule: a road at spline resolution 2 is bumpy and at 64 is smooth *at
identical polygon count* — it is vertex placement along a smooth analytic curve,
not vertex count, that removes false bumps.

### Format: OpenDRIVE, not Lanelet2
OpenDRIVE (.xodr) stores exactly the analytic model needed: reference line as
`<line>/<arc>/<spiral>/<paramPoly3>`, `<elevationProfile>` cubic, **`<lateralProfile>
<superelevation>` (camber, in rad)**, lane `<width>` as a cubic in sOffset, lane
`<height inner outer>` for kerb step, and lane `type` (`driving / border / sidewalk
/ curb / shoulder`) which is `on_track` and the boundary selector for free.

Lanelet2 (BSD-3, active) is **wrong for this**: no curve primitives (curvature
becomes finite differences over noisy polyline vertices), no width attribute
(implicit as distance between two boundaries), **no superelevation concept at all**.
OpenDRIVE→Lanelet2 is straightforward; the reverse is "much harder" because you are
inventing curve primitives that were never there. **Anything routed via Lanelet2 has
already thrown camber away.**

### No converter's output is usable as-is
| Tool | The problem |
|---|---|
| **SUMO `netconvert`** (EPL-2.0, v1.27.1 2026-06) | Writes `<lateralProfile/>` **unconditionally empty**. Lane width constant (`b=c=d=0`). But: real `<elevation>` cubics, `--heightmap.geotiff` DEM ingestion (**needs GDAL**), and by far the **best junction builder** (`<junction>/<connection>/<laneLink>`) |
| CommonRoad Scenario Designer (GPL-3.0) | Writes **both** elevation and lateral profiles empty — `set_child_of_road(ELEVATION_PROFILE_TAG)` with nothing added. Vertices are Nx2. ~12 months quiet |
| osm2xodr (GPL-3.0) | **Dead** (2022). Reads its heightmap from channel 0 of an 8-bit RGB → stair-stepped elevation. Width hardcoded 4.0 m |
| CARLA `Osm2Odr` | Vendored SUMO fork; `elevation_layer_height` is an OSM `layer` heuristic, **not a DEM**. Standalone mesh mode documents "Lateral slope — not integrated" (**no camber**) |
| libOpenDRIVE (Apache-2.0) | `get_surface_pt(s,t,&normal)` gives camber; `get_lane_border_line(...)` gives the **rangefinder boundary polyline**. But **no `get_curvature()`** and no map-level XY→road lookup — build your own index |
| **scenariogeneration / pyodrx** (MPL-2.0, v0.16.6 2026-07) | **Healthiest OpenDRIVE writer.** Targets 1.7.1, has `ElevationCalculator` / `adjust_elevations()`. **This is the camber injection point** |
| OSM2World, BlenderGIS, RoadRunner | Visual/authoring value only — worth zero to a pixel-free sim |

**Recipe: use `netconvert` for topology and junctions only, then rewrite the
geometry yourself with `scenariogeneration`.**
`--junctions.join` and `--geometry.remove` are **mandatory** on dense urban OSM or
you get phantom micro-junctions every few metres.

### Junction surfaces — sidestepped, not solved
ASAM concede the spec itself is the problem: connecting roads "only need to describe
connectivity rather than junction boundaries", and are the only roads in OpenDRIVE
with overlapping surfaces. In the **analytic** domain this is not z-fighting, it is
query ambiguity — answered by `RM_GetNumberOfRoadsOverlapping`. For `on_track`
inside a junction take the union of drivable lane areas; for the boundary polygon
use osm2streets' intersection polygon.

---

## 5. Random closed circuit

This is the ***k*-length round trip (KRT)** problem. **NP-hard.**
Lewis & Corcoran (Cardiff), *SN Computer Science* 5:868 (2024),
doi:10.1007/s42979-024-03223-3 — open access, **code+data: zenodo.org/doi/10.5281/zenodo.8154412**.
Also *Journal of Heuristics* 28:259–285 (2022).

- **Cleanest construction for a true lap**: pick target t at graph distance ≈ k/2,
  compute a minimum-cost pair of **edge-disjoint s–t paths** (Suurballe/Bhandari),
  concatenate one forward and one reversed. Zero repetition, guaranteed cycle.
- **The trick home-grown implementations miss**: GraphHopper's `RoundTripRouting`
  uses `AvoidEdgesWeighting.setEdgePenaltyFactor(5)` — each routed leg's edges are
  penalised 5× on later legs. Its `roundTripPointCount = 2 + distance/50000`, so a
  10 km loop uses just 2 intermediate points.
- **Out-and-back removal**: treat the lap's arcs as an undirected simple graph and
  iteratively delete degree-1 vertices (excluding start). Four lines; kills the
  dead-end-hairpin failure mode.
- **Sample the start vertex from inside a biconnected component** — Lewis warns
  road graphs are not necessarily biconnected.
- **Don't use `nx.minimum_cycle_basis`** — O(m²n), ~7×10¹² ops on a city network.
  `nx.simple_cycles(length_bound=k)` bounds **hop count, not metres** — useless.
- **Turn restrictions need an edge-expanded graph** (directed segment → node, legal
  turn → edge). **OSMnx does not import restriction relations at all** — they are
  relations, not way tags. 57 exist in the bbox.
- **Smoothing**: fit a closed **G2 clothoid spline** (`pyclothoids`, MIT) or a
  periodic cubic B-spline (`splprep(..., per=1)`). Clothoids are right because
  curvature is linear in arc length — exactly OpenDRIVE's `<spiral>`.
  **Put start/finish at the point of maximum local radius** (longest straight), as
  real circuits do, so any residual discontinuity is invisible.
- **Do not simplify with Douglas-Peucker** — its error metric is positional, but
  curvature is a second derivative; sub-metre positional error turns a sweep into a
  chord plus a kink. Resample to uniform arc length (2–5 m) instead.
- **Lap length**: FIA Appendix O Grade 1 is min 3.5 km, recommended max 7 km
  (figures from secondary reporting — the official PDF 504'd). A 4–8 km target is sensible.
- **Validate against** `TUMFTM/racetrack-database` (LGPL-3.0): 25 real circuits as
  `[x, y, w_tr_right, w_tr_left]`, centrelines from OSM and **widths extracted from
  satellite imagery by an image-processing algorithm** — a proven precedent for
  exactly the width problem in §1. Hand off to `TUMFTM/global_racetrajectory_optimization`.

**No existing project picks a random closed loop of real roads and emits it as a
race circuit.** Both halves have mature components; nobody has bolted them together.

---

## 6. The fidelity ceiling, stated honestly

| Quantity | Recoverable? |
|---|---|
| Centreline geometry, curvature | **Yes, good** — OSM's strength |
| Junction topology | **Yes** via SUMO |
| Longitudinal grade | **Yes** from 1 m LiDAR — but only ~19 m of macro ramp survives the required 70 m smoothing |
| **Carriageway width** | **Weak** — ±15–20 % inferred, or hand-traced |
| **Camber / superelevation** | **Synthesised** from DMRB CD 109. Contributes ~5 % of cornering accel and lane-averages to ≈ 0 |
| Racing-circuit banking (Zandvoort 18°, Daytona 31°) | **No, ever** — public roads cap near 4° |
| Surface micro-relief → `airborne` | **No.** Needs ASAM OpenCRG and a centimetre road scan. A spline-fitted alignment is smooth by construction |
| Kerb geometry | **No** — 33 mm recovered against a real 100–125 mm upstand |

### The uncomfortable consequence for the generalisation test
SPEC §4 calls London "the honest generalisation test, because nothing about London
is in the training set". That is true — but the measurements say London is also
**geometrically less demanding than Mario Kart**: flat (0.9 % macro gradient, ~4 %
worst street), no banking, no jumps, `airborne` permanently zero, camber ≈ 0 after
lane-averaging. MK64 → London is a **narrower** domain, not a wider one.

So a policy may clear London easily while proving little about generalisation, and
the interesting signal moves to the stage-2 confidence trace — is it *correctly*
confident on geometry that is unlike Nintendo's but genuinely easier? Worth deciding
before treating a completed London lap as the project's success criterion.

---

## 7. Recommended pipeline

1. **Data**: Geofabrik `greater-london-latest.osm.pbf` (128.7 MB) + one EA WCS call
   (16 MB). Both keyless. Read the PBF with `pyrosm` (Cython, no GDAL).
2. **Loop selection** on an edge-expanded OSMnx drive graph: edge-disjoint-path
   construction, out-and-back removal, heading-change rejection (bearings from
   geometry **endpoints**, not the u→v chord), score on |k−L| / repetition / min radius.
3. **Geometry**: resample to uniform arc length, fit a closed G2 clothoid spline,
   seam at the longest straight, verify κ(s) continuous and κ(0)=κ(L).
4. **Elevation**: sample the DTM along the centreline, **smooth ≥70 m**, fit z(s),
   clamp grade, **hardcode the Westminster Bridge deck**, despike sub-zero cells.
5. **Camber**: synthesise from κ(s) via DMRB CD 109, smoothed runoff.
6. **Emit .xodr** with `scenariogeneration` — real `<spiral>`, populated
   `<elevation>` and `<superelevation>`, real `<width>`.
7. **Query at runtime**: esminiRMLib via ctypes for the observation vector;
   libOpenDRIVE `get_lane_border_line` or osm2streets polygons for rangefinder
   boundary; single-track bicycle model with Pacejka tyres on top.

**Only install friction: GDAL**, needed for `netconvert --heightmap.geotiff`.
Not needed to read the WCS TIFF (PIL mode `F`).

---

## 8. Unverified — do not treat as settled
- FIA Appendix O figures are from secondary reporting; the official PDF 504'd twice.
- Whether esmini parses `<surface><CRG>`.
- CommonRoad Scenario Designer version split (docs say 0.9.3-beta, PyPI says 0.8.5).
- libOpenDRIVE commits after 2025-12-31 (GitHub API rate-limited; dates from atom feeds).
- OpenTwinMap (arXiv 2511.21925) repo URL and licence.
