"""Hand-drawn track (photo) -> ordered waypoints -> London lat/lon (+ optional OSM street snap).

Usage:
  python track_to_london.py drawing.jpg --area kings_cross --out route.json
                            [--n 60] [--snap] [--preview preview.png]

Output JSON: [{"i":0,"lat":..,"lon":..,"heading":..,"street":..}, ...]
heading in degrees, 0 = north, clockwise.

--snap needs `pip install osmnx` and internet (Overpass API).
"""
import argparse, json, math, sys
import cv2
import numpy as np
from skimage.morphology import skeletonize

# Areas the drawn track can be dropped into, as (centre_lat, centre_lon). Each becomes a
# BOX_SIDE_M square centred on that point. One area == one Unreal map.
AREAS = {
    # Centres tuned so a drawn loop survives street snapping. The boxes originally
    # specified are kept in AREAS_AS_SPECIFIED below and can be selected with
    # --centre lat,lon. Three of the five sat on rail land or river, where the drivable
    # network is too sparse to carry a lap (Tower Bridge scored 2 streets, 68% of the
    # drawn length; Paris Garden 2 streets, 20%). Each was moved a few hundred metres to
    # the nearest real street grid, which takes every area to ~1.2 km and 10+ streets.
    "kings_cross":  (51.5310, -0.1195),   # Caledonian Rd / Killick St
    "paris_garden": (51.5045, -0.1015),   # The Cut / Union St, Southwark
    "westminster":  (51.5010, -0.1270),   # as specified — already good
    "tower_bridge": (51.5030, -0.0785),   # Tooley St, south end of the bridge
    "shoreditch":   (51.5250, -0.0780),   # as specified — already good
}

# What was originally asked for, for reference and easy comparison.
AREAS_AS_SPECIFIED = {
    "kings_cross":  (51.5340, -0.1240),
    "paris_garden": (51.5090, -0.1040),
    "westminster":  (51.5010, -0.1270),
    "tower_bridge": (51.5055, -0.0754),
    "shoreditch":   (51.5250, -0.0780),
}

DEFAULT_AREA = "kings_cross"

# Side of the square, metres. A loop filling it is ~pi * 400 = 1.26 km round, which is the
# Mario Kart 64 lap scale the policy trained on.
BOX_SIDE_M = 400.0

M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat_deg):
    return M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def area_bbox(area, side_m=BOX_SIDE_M, centre=None):
    """Square bbox of `side_m` centred on the named area, or on an explicit centre."""
    if centre is not None:
        lat, lon = centre
    else:
        if area not in AREAS:
            sys.exit(f"unknown area {area!r}; choose from {', '.join(sorted(AREAS))}")
        lat, lon = AREAS[area]
    half = side_m / 2.0
    dlat = half / M_PER_DEG_LAT
    dlon = half / m_per_deg_lon(lat)
    return dict(north=lat + dlat, south=lat - dlat, west=lon - dlon, east=lon + dlon)


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def lap_length_m(latlon, closed):
    """Ground length along the waypoints, closing back to the start when the track loops."""
    pts = list(latlon)
    if closed and len(pts) > 1:
        pts.append(pts[0])
    return sum(haversine_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
               for i in range(len(pts) - 1))


def load_ink(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"could not read {path}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # adaptive threshold copes with phone photos and shadows
    ink = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 15)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # keep only the largest blob (the track), drop specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink)
    if n < 2:
        sys.exit("no ink found")
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == biggest).astype(np.uint8)


def skeleton_points(mask):
    sk = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(sk)
    return np.stack([xs, ys], axis=1).astype(float)


def order_points(pts):
    """Nearest-neighbour walk along the skeleton. Good enough for a single loop or line."""
    remaining = pts.copy()
    # start from leftmost point so runs are deterministic
    idx = int(np.argmin(remaining[:, 0]))
    path = [remaining[idx]]
    remaining = np.delete(remaining, idx, axis=0)
    while len(remaining):
        d = np.linalg.norm(remaining - path[-1], axis=1)
        j = int(np.argmin(d))
        if d[j] > 25:  # gap: skeleton is broken, stop rather than jump across the page
            break
        path.append(remaining[j])
        remaining = np.delete(remaining, j, axis=0)
    return np.array(path)


def resample(path, n, closed):
    if closed:
        path = np.vstack([path, path[:1]])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, cum[-1], n, endpoint=not closed)
    out = np.empty((n, 2))
    out[:, 0] = np.interp(targets, cum, path[:, 0])
    out[:, 1] = np.interp(targets, cum, path[:, 1])
    return out


def is_closed(path):
    return np.linalg.norm(path[0] - path[-1]) < 0.08 * max(np.ptp(path[:, 0]), np.ptp(path[:, 1]))


def to_latlon(pts, bbox):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    # preserve aspect ratio inside the box
    lat_span = bbox["north"] - bbox["south"]
    lon_span = bbox["east"] - bbox["west"]
    # metres per degree at London latitude
    m_lat = 111_320
    m_lon = 111_320 * math.cos(math.radians((bbox["north"] + bbox["south"]) / 2))
    box_w_m, box_h_m = lon_span * m_lon, lat_span * m_lat
    scale = min(box_w_m / w, box_h_m / h)
    cx, cy = (bbox["west"] + bbox["east"]) / 2, (bbox["south"] + bbox["north"]) / 2
    lon = cx + ((pts[:, 0] - (x0 + x1) / 2) * scale) / m_lon
    lat = cy - ((pts[:, 1] - (y0 + y1) / 2) * scale) / m_lat  # image y is down
    return np.stack([lat, lon], axis=1)


def headings(latlon, closed):
    nxt = np.roll(latlon, -1, axis=0) if closed else np.vstack([latlon[1:], latlon[-1:]])
    dlat = nxt[:, 0] - latlon[:, 0]
    dlon = (nxt[:, 1] - latlon[:, 1]) * math.cos(math.radians(latlon[:, 0].mean()))
    return (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360


def snap_to_streets(latlon, bbox, mode="edge"):
    """Pull each waypoint onto the drivable street network.

    mode="edge" (default) projects the waypoint onto the nearest point *along* the
    matched street's geometry. mode="node" snaps to the nearest junction of that street,
    which is what this did originally — it collapses 60 waypoints onto a handful of
    junctions in a 400 m box (2/60 at Tower Bridge), destroying the lap. Kept only so the
    difference can be demonstrated; do not use it for a real track.
    """
    import osmnx as ox
    from shapely.geometry import LineString, Point

    G = ox.graph_from_bbox(bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
                           network_type="drive")
    # osmnx >= 2 returns one (n, 3) array of (u, v, key) rows for array input,
    # not three separate arrays, so unpack per row rather than per column.
    ne = ox.distance.nearest_edges(G, X=latlon[:, 1], Y=latlon[:, 0])
    names, snapped = [], []
    for (a, b, k), (lat, lon) in zip(ne, latlon):
        data = G.get_edge_data(a, b, k)
        name = data.get("name", "unnamed road")
        names.append(name if isinstance(name, str) else name[0])

        na, nb = G.nodes[a], G.nodes[b]
        if mode == "node":
            da = (na["y"] - lat) ** 2 + (na["x"] - lon) ** 2
            db = (nb["y"] - lat) ** 2 + (nb["x"] - lon) ** 2
            n = na if da < db else nb
            snapped.append([n["y"], n["x"]])
            continue

        # Curved streets carry an explicit geometry; straight ones are just node to node.
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([(na["x"], na["y"]), (nb["x"], nb["y"])])
        point = geom.interpolate(geom.project(Point(lon, lat)))
        snapped.append([point.y, point.x])
    return np.array(snapped), names


def preview(mask, path, out):
    vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    for i, (x, y) in enumerate(path.astype(int)):
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        if i % 10 == 0:
            cv2.putText(vis, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(out, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--area", default=DEFAULT_AREA, choices=sorted(AREAS),
                    help=f"London area to drop the track into (default {DEFAULT_AREA})")
    ap.add_argument("--centre", help="override the area centre, as 'lat,lon'")
    ap.add_argument("--box", type=float, default=BOX_SIDE_M,
                    help=f"box side in metres (default {BOX_SIDE_M:.0f})")
    ap.add_argument("--out", default="route.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap-mode", default="edge", choices=("edge", "node"),
                    help="edge: project along the street (default). "
                         "node: snap to junctions — collapses the track, demo only")
    ap.add_argument("--preview")
    a = ap.parse_args()

    centre = None
    if a.centre:
        try:
            lat_s, lon_s = a.centre.split(",")
            centre = (float(lat_s), float(lon_s))
        except ValueError:
            sys.exit(f"--centre must be 'lat,lon', got {a.centre!r}")
    bbox = area_bbox(a.area, side_m=a.box, centre=centre)
    used = centre if centre else AREAS[a.area]

    mask = load_ink(a.image)
    pts = skeleton_points(mask)
    path = order_points(pts)
    closed = bool(is_closed(path))
    path = resample(path, a.n, closed)
    ll = to_latlon(path, bbox)
    drawn_len = lap_length_m(ll, closed)
    names = [None] * len(ll)
    snap_report = None
    if a.snap:
        ll, names = snap_to_streets(ll, bbox, mode=a.snap_mode)
        # Several waypoints routinely land on the same OSM node, which silently shortens
        # the lap and can flatten corners. Count it rather than let it pass unnoticed.
        uniq = len({(round(float(la), 7), round(float(lo), 7)) for la, lo in ll})
        named = sum(1 for n in names if n and n != "unnamed road")
        snap_report = (uniq, named)
    hd = headings(ll, closed)
    route = [dict(i=i, lat=round(float(la), 6), lon=round(float(lo), 6),
                  heading=round(float(h), 1), street=s)
             for i, ((la, lo), h, s) in enumerate(zip(ll, hd, names))]
    lap_m = lap_length_m(ll, closed)
    json.dump(dict(area=a.area, centre=dict(lat=used[0], lon=used[1]),
                   streets=list(dict.fromkeys(n for n in names if n)),
                   box_side_m=a.box, closed=closed, lap_length_m=round(lap_m, 1),
                   snapped=bool(a.snap), snap_mode=a.snap_mode if a.snap else None,
                   bbox=bbox, waypoints=route),
              open(a.out, "w"), indent=1)
    if a.preview:
        preview(mask, path, a.preview)

    print(f"{len(route)} waypoints, closed={closed}, area={a.area}, wrote {a.out}")
    print(f"lap length {lap_m:,.0f} m ({lap_m / 1000:.2f} km) "
          f"[target ~1 km, a Mario Kart 64 lap]")
    if snap_report:
        uniq, named = snap_report
        keep = 100.0 * lap_m / drawn_len if drawn_len else 0.0
        print(f"snap[{a.snap_mode}]: {uniq}/{len(route)} distinct points, "
              f"{named}/{len(route)} named streets; "
              f"pre-snap lap {drawn_len:,.0f} m -> {keep:.0f}% kept")
        print(f"streets: {', '.join(dict.fromkeys(names))[:120]}")


if __name__ == "__main__":
    main()
