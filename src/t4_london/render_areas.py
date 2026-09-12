"""Render the per-area routes side by side as one presentation PNG.

    python render_areas.py --tracks ../../tracks --out ../../runs/areas.png

Each panel draws the real drivable street network for that 400 m box in grey, the lap
on top, a dot at the start, a 100 m scale bar and a north arrow. Street geometry is
fetched from OSM (same source as --snap), so what you see is what the car will drive.
"""
import argparse, json, os

import cv2
import numpy as np

AREA_ORDER = ["kings_cross", "paris_garden", "westminster", "tower_bridge", "shoreditch"]

MAP = 430
PAD = 30
HEAD = 86
FOOT = 54
PANEL_W = MAP + 2 * PAD
PANEL_H = HEAD + MAP + 2 * PAD + FOOT

BG      = (252, 251, 249)
PLATE   = (255, 255, 255)
BOXLINE = (206, 202, 196)
STREET  = (214, 210, 204)
ROUTE   = (38, 104, 226)      # BGR -> warm orange-red
ROUTE_H = (150, 190, 250)     # halo
START   = (86, 176, 74)       # BGR -> green
INK     = (34, 32, 30)
MUTED   = (128, 124, 118)
GOOD    = (86, 150, 60)
WARN    = (60, 110, 210)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT2 = cv2.FONT_HERSHEY_SIMPLEX


def to_px(lat, lon, bbox):
    fx = (np.asarray(lon) - bbox["west"]) / (bbox["east"] - bbox["west"])
    fy = (bbox["north"] - np.asarray(lat)) / (bbox["north"] - bbox["south"])
    x = PAD + fx * MAP
    y = HEAD + PAD + fy * MAP
    return np.stack([x, y], -1)


def fetch_streets(bbox):
    """Drivable street geometry inside the box, as lists of (lat, lon)."""
    try:
        import osmnx as ox
        from shapely.geometry import LineString
        G = ox.graph_from_bbox(bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
                               network_type="drive")
    except Exception as exc:
        print(f"  (streets unavailable: {type(exc).__name__})")
        return []
    out = []
    for a, b, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([(G.nodes[a]["x"], G.nodes[a]["y"]),
                               (G.nodes[b]["x"], G.nodes[b]["y"])])
        xs, ys = geom.xy
        out.append(list(zip(ys, xs)))
    return out


def panel(route, streets):
    img = np.full((PANEL_H, PANEL_W, 3), BG, np.uint8)
    bbox = route["bbox"]
    x0, y0, x1, y1 = PAD, HEAD + PAD, PAD + MAP, HEAD + PAD + MAP
    cv2.rectangle(img, (x0, y0), (x1, y1), PLATE, -1)

    for seg in streets:
        pts = to_px([p[0] for p in seg], [p[1] for p in seg], bbox).astype(np.int32)
        cv2.polylines(img, [pts], False, STREET, 2, cv2.LINE_AA)

    wps = route["waypoints"]
    pts = to_px([w["lat"] for w in wps], [w["lon"] for w in wps], bbox).astype(np.int32)
    closed = route.get("closed", True)
    cv2.polylines(img, [pts], closed, ROUTE_H, 7, cv2.LINE_AA)
    cv2.polylines(img, [pts], closed, ROUTE, 3, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 8, START, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 8, PLATE, 2, cv2.LINE_AA)

    cv2.rectangle(img, (x0, y0), (x1, y1), BOXLINE, 1)

    # 100 m scale bar
    span_m = route.get("box_side_m", 400.0)
    bar = int(MAP * (100.0 / span_m))
    by = y1 - 16
    cv2.line(img, (x0 + 14, by), (x0 + 14 + bar, by), INK, 2, cv2.LINE_AA)
    cv2.putText(img, "100 m", (x0 + 14, by - 7), FONT2, 0.40, INK, 1, cv2.LINE_AA)

    # north arrow
    nx, ny = x1 - 22, y0 + 30
    cv2.arrowedLine(img, (nx, ny), (nx, ny - 18), INK, 2, cv2.LINE_AA, tipLength=0.45)
    cv2.putText(img, "N", (nx - 5, ny + 14), FONT2, 0.42, INK, 1, cv2.LINE_AA)

    name = route["area"].replace("_", " ").title()
    lap = route.get("lap_length_m", 0.0)
    streets_named = route.get("streets") or []
    cv2.putText(img, name, (PAD, 32), FONT, 0.70, INK, 1, cv2.LINE_AA)
    cv2.putText(img, f"{lap/1000:.2f} km lap", (PAD, 57), FONT, 0.56, ROUTE, 1, cv2.LINE_AA)
    head = ", ".join(streets_named[:3])
    if len(head) > 44:
        head = head[:41] + "..."
    cv2.putText(img, head or "-", (PAD, 76), FONT2, 0.40, MUTED, 1, cv2.LINE_AA)

    uniq = len({(round(w["lat"], 7), round(w["lon"], 7)) for w in wps})
    frac = uniq / max(len(wps), 1)
    cv2.putText(img, f"{uniq}/{len(wps)} distinct waypoints  -  {len(streets_named)} streets",
                (PAD, HEAD + MAP + 2 * PAD + 22), FONT2, 0.42,
                GOOD if frac > 0.7 else WARN, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="../../tracks")
    ap.add_argument("--out", default="../../runs/areas.png")
    ap.add_argument("--no-streets", action="store_true")
    ap.add_argument("--title", default="One hand-drawn track, five 400 m London boxes")
    ap.add_argument("--subtitle",
                    default="waypoints snapped along real drivable streets (OSM)  -  "
                            "target ~1 km, a Mario Kart 64 lap")
    a = ap.parse_args()

    panels = []
    for area in AREA_ORDER:
        path = os.path.join(a.tracks, f"route_{area}.json")
        if not os.path.exists(path):
            print(f"skipping {area}: no {path}")
            continue
        route = json.load(open(path))
        print(f"  {area}")
        streets = [] if a.no_streets else fetch_streets(route["bbox"])
        panels.append(panel(route, streets))

    if not panels:
        raise SystemExit("no routes found")

    strip = np.hstack(panels)
    bar = np.full((78, strip.shape[1], 3), BG, np.uint8)
    cv2.putText(bar, a.title, (PAD, 34), FONT, 0.80, INK, 1, cv2.LINE_AA)
    cv2.putText(bar, a.subtitle, (PAD, 58), FONT2, 0.46, MUTED, 1, cv2.LINE_AA)
    out = np.vstack([bar, strip])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cv2.imwrite(a.out, out)
    print(f"wrote {a.out} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
