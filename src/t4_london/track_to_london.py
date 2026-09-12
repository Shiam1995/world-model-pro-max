"""Hand-drawn track (photo) -> ordered waypoints -> London lat/lon (+ optional OSM street snap).

Usage:
  python track_to_london.py drawing.jpg --out route.json [--n 60] [--snap] [--preview preview.png]

Output JSON: [{"i":0,"lat":..,"lon":..,"heading":..,"street":..}, ...]
heading in degrees, 0 = north, clockwise.

--snap needs `pip install osmnx` and internet (Overpass API).
"""
import argparse, json, math, sys
import cv2
import numpy as np
from skimage.morphology import skeletonize

# Default London box: Westminster / St James's / Whitehall. Change freely.
BBOX = dict(north=51.5090, south=51.4980, west=-0.1440, east=-0.1200)


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


def snap_to_streets(latlon, bbox):
    import osmnx as ox
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
        # snap to nearest node of the edge
        na, nb = G.nodes[a], G.nodes[b]
        da = (na["y"] - lat) ** 2 + (na["x"] - lon) ** 2
        db = (nb["y"] - lat) ** 2 + (nb["x"] - lon) ** 2
        n = na if da < db else nb
        snapped.append([n["y"], n["x"]])
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
    ap.add_argument("--out", default="route.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--preview")
    a = ap.parse_args()

    mask = load_ink(a.image)
    pts = skeleton_points(mask)
    path = order_points(pts)
    closed = bool(is_closed(path))
    path = resample(path, a.n, closed)
    ll = to_latlon(path, BBOX)
    names = [None] * len(ll)
    if a.snap:
        ll, names = snap_to_streets(ll, BBOX)
    hd = headings(ll, closed)
    route = [dict(i=i, lat=round(float(la), 6), lon=round(float(lo), 6),
                  heading=round(float(h), 1), street=s)
             for i, ((la, lo), h, s) in enumerate(zip(ll, hd, names))]
    json.dump(dict(closed=closed, bbox=BBOX, waypoints=route), open(a.out, "w"), indent=1)
    if a.preview:
        preview(mask, path, a.preview)
    print(f"{len(route)} waypoints, closed={closed}, wrote {a.out}")


if __name__ == "__main__":
    main()
