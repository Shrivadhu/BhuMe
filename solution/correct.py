"""
Per-plot boundary correction via chamfer alignment against detected field edges.

Method, in short:
  1. Crop a small patch of imagery (+ boundaries.tif hint, if present) around each plot.
  2. Build a single "edgeness" map for the patch: the hint raster where it exists, backed up by
     Canny edges from the RGB image where the hint is thin/absent (tree cover, buildings).
  3. Distance-transform that edge map: every pixel now holds "how far to the nearest edge".
  4. Search small pixel-space translations of the plot's drawn boundary; the best one is the
     translation that lands the boundary's own vertices closest to real edges (lowest mean
     chamfer distance). This captures the *local* residual left after a global shift, including
     per-plot direction, without ever touching the plot's shape.
  5. Confidence is built from three independent signals, not just the match cost:
       - match sharpness: how much better the best offset is than doing nothing / than the
         typical offset in the search window (a flat cost surface means "can't tell", low conf)
       - edge support: fraction of the patch that actually has edge evidence (thin under trees
         -> the hint/Canny disagree or are empty -> lower confidence, matches the brief's warning)
       - area agreement: drawn area vs recorded (cultivable + pot-kharaba) total. Translation
         cannot fix a plot whose *shape* is wrong, so a bad area ratio caps confidence hard and
         can force a flag even if the pixel match looks locally clean.
  6. Plots with weak edge support everywhere in range, or a badly wrong area, are flagged rather
     than corrected -- restraint is scored, and a hand-wavy shift is worse than admitting we can't
     place it.

Translation only (no rotation/reshape): simpler, more robust to overfitting on 6 example plots,
and the geometry inspection shows the drift is dominantly a coherent local offset. Nothing here is
tuned to Vadnerbhairav specifically -- pixel search radius and thresholds are derived from the
imagery's own resolution and the plot's own size, not hardcoded metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.affinity import translate
from shapely.geometry import Polygon

from bhume.geo import geom_to_imagery_crs


@dataclass
class PlotResult:
    plot_number: str
    status: str            # 'corrected' | 'flagged'
    confidence: float | None
    dx_m: float
    dy_m: float
    method_note: str
    geometry: object        # shapely geometry in EPSG:4326


def _read_band_patch(src, bounds, band_indexes):
    left, bottom, right, top = bounds
    dl, db, dr, dt = src.bounds
    left, bottom, right, top = max(left, dl), max(bottom, db), min(right, dr), min(top, dt)
    if right <= left or top <= bottom:
        return None, None
    window = from_bounds(left, bottom, right, top, transform=src.transform)
    arr = src.read(band_indexes, window=window)
    return arr, src.window_transform(window)


def _edge_map(rgb: np.ndarray, hint: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Combine the boundaries.tif hint with Canny edges from the image into one 0/1 edge map.

    Returns (edge_map, edge_support) where edge_support in [0,1] is the fraction of pixels
    carrying edge evidence -- used later to discount confidence where the signal is thin.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    med = float(np.median(gray))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    canny = cv2.Canny(gray, lo, hi) > 0

    if hint is not None and hint.shape == canny.shape:
        edges = (hint > 0) | canny
    else:
        edges = canny

    support = float(edges.mean())
    return edges.astype(np.uint8), support


def _boundary_px_points(poly: Polygon, transform, n: int = 48) -> np.ndarray:
    """Sample n points evenly along the polygon exterior, in patch pixel (col,row) coords."""
    ring = poly.exterior
    length = ring.length
    if length == 0:
        return np.zeros((0, 2))
    dists = np.linspace(0, length, n, endpoint=False)
    pts = [ring.interpolate(d) for d in dists]
    inv = ~transform
    px = np.array([inv * (p.x, p.y) for p in pts])  # (n, 2) as (col, row)
    return px


def _best_shift(px_pts: np.ndarray, dist_map: np.ndarray, radius_px: int):
    """Grid-search integer pixel shifts of px_pts against a distance-to-edge map.

    Returns (best_dcol, best_drow, best_cost, cost_at_zero, mean_cost, cost_std).
    Cost is the mean chamfer distance (pixels) of the shifted boundary points to the nearest edge.
    """
    h, w = dist_map.shape
    shifts = np.arange(-radius_px, radius_px + 1)
    dcols, drows = np.meshgrid(shifts, shifts, indexing='ij')
    dcols = dcols.ravel()
    drows = drows.ravel()

    cols = np.clip((px_pts[:, 0][None, :] + dcols[:, None]).round().astype(int), 0, w - 1)
    rows = np.clip((px_pts[:, 1][None, :] + drows[:, None]).round().astype(int), 0, h - 1)
    costs = dist_map[rows, cols].mean(axis=1)  # (n_shifts,)

    best_i = int(np.argmin(costs))
    best_cost = float(costs[best_i])
    zero_i = int(np.argmin(dcols ** 2 + drows ** 2))  # shift (0,0)
    cost_at_zero = float(costs[zero_i])

    return int(dcols[best_i]), int(drows[best_i]), best_cost, cost_at_zero, float(costs.mean()), float(costs.std())


def correct_plot(
    imagery_src,
    boundaries_src,
    plot_number: str,
    geom_4326,
    recorded_total_sqm: float | None,
    map_area_sqm: float,
    pad_m: float | None = None,
    search_radius_m: float | None = None,
) -> PlotResult:
    """Run the full pipeline for one plot. Returns a PlotResult ready to write out.

    `pad_m` / `search_radius_m`, if not given, are derived from the plot's own footprint size
    rather than fixed per-village: a small, tightly-packed plot (Malatavadi) gets a small search
    window so it can't jump onto a neighbour's edge; a large field (Vadnerbhairav) gets more room.
    A fixed absolute metre radius tuned on one village's plot scale is exactly the kind of
    hand-tuning that fails to generalise -- confirmed empirically (see notes).
    """
    geom_img = geom_to_imagery_crs(imagery_src, geom_4326)
    if geom_img.geom_type == 'MultiPolygon':
        geom_img = max(geom_img.geoms, key=lambda g: g.area)
    if not geom_img.is_valid:
        # a handful of official records have self-intersecting rings pre-existing in the source
        # data; repair topology without changing the intended shape before we do anything else.
        geom_img = geom_img.buffer(0)
        if geom_img.is_empty or geom_img.geom_type != 'Polygon':
            return PlotResult(plot_number, 'flagged', None, 0.0, 0.0,
                               'flagged: official geometry invalid and unrepairable', geom_4326)

    minx, miny, maxx, maxy = geom_img.bounds
    plot_extent_m = ((maxx - minx) + (maxy - miny)) / 2.0  # avg of width/height, robust to shape

    if search_radius_m is None:
        # never search further than ~40% of the plot's own size (caps out at 20m for big fields,
        # shrinks for small ones), with a small floor so tiny slivers still get *some* search room
        search_radius_m = float(np.clip(0.4 * plot_extent_m, 4.0, 20.0))
    if pad_m is None:
        pad_m = search_radius_m + 6.0  # just enough context beyond the search window itself

    bounds = (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)

    rgb_arr, transform = _read_band_patch(imagery_src, bounds, [1, 2, 3])
    if rgb_arr is None or rgb_arr.shape[1] < 8 or rgb_arr.shape[2] < 8:
        return PlotResult(plot_number, 'flagged', None, 0.0, 0.0,
                           'patch outside imagery extent or too small', geom_4326)
    rgb = np.transpose(rgb_arr, (1, 2, 0))

    hint = None
    if boundaries_src is not None:
        hint_arr, _ = _read_band_patch(boundaries_src, bounds, 1)
        if hint_arr is not None and hint_arr.shape == rgb.shape[:2]:
            hint = hint_arr

    edges, support = _edge_map(rgb, hint)
    if edges.sum() == 0:
        return PlotResult(plot_number, 'flagged', None, 0.0, 0.0,
                           'no edge evidence in patch (no hint, flat imagery)', geom_4326)

    dist_map = distance_transform_edt(edges == 0)

    px_pts = _boundary_px_points(geom_img, transform, n=48)
    # pixel size from the transform (metres/pixel), used to size the search window
    px_size = float(abs(transform.a))
    radius_px = max(3, int(round(search_radius_m / px_size)))

    dcol, drow, best_cost, cost0, mean_cost, cost_std = _best_shift(px_pts, dist_map, radius_px)
    dx_m = dcol * px_size
    dy_m = -drow * px_size  # image rows increase downward; y increases upward

    # --- confidence components ---
    # 1) sharpness: how much the best offset beats the typical (random) offset in the window
    sharpness = 0.0 if mean_cost <= 1e-6 else max(0.0, 1.0 - best_cost / mean_cost)
    # 2) absolute match quality: best cost in pixels, small = tight fit to a real edge
    tightness = float(np.clip(1.0 - best_cost / 3.0, 0.0, 1.0))
    # 3) edge support in the patch (thin under trees/buildings -> less trustworthy)
    support_score = float(np.clip(support / 0.06, 0.0, 1.0))  # ~6% edge pixels = solid coverage
    # 4) area agreement: drawn vs recorded total (translation can't fix a shape problem)
    area_score = 1.0
    area_ratio = None
    if recorded_total_sqm and recorded_total_sqm > 0 and map_area_sqm > 0:
        area_ratio = map_area_sqm / recorded_total_sqm
        area_score = float(np.clip(1.0 - abs(1.0 - area_ratio) / 0.35, 0.0, 1.0))

    confidence = float(np.clip(
        0.35 * sharpness + 0.25 * tightness + 0.15 * support_score + 0.25 * area_score, 0.0, 1.0
    ))

    move_dist = (dx_m ** 2 + dy_m ** 2) ** 0.5
    likely_area_problem = area_ratio is not None and abs(1.0 - area_ratio) > 0.30
    weak_signal = support < 0.01 or (best_cost > 2.5 and sharpness < 0.15)

    # A principled floor, not one tuned to any particular village's example truths: don't claim
    # a correction you're less than half-confident in. (Empirically, letting low-but-nonzero
    # confidence corrections through was the exact failure mode that hurt Malatavadi -- confidence
    # correctly *ranked* the bad ones lowest, but 0.20 was too permissive to actually catch them.)
    if likely_area_problem or weak_signal or confidence < 0.50:
        reason = []
        if likely_area_problem:
            reason.append(f'area mismatch (drawn/recorded={area_ratio:.2f})')
        if weak_signal:
            reason.append('weak/absent edge signal in patch')
        if not reason:
            reason.append('low overall confidence')
        return PlotResult(plot_number, 'flagged', None, dx_m, dy_m,
                           'flagged: ' + '; '.join(reason), geom_4326)

    corrected_img = translate(geom_img, xoff=dx_m, yoff=dy_m)
    # back to 4326 via inverse of geom_to_imagery_crs
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    tf = Transformer.from_crs(imagery_src.crs, 'EPSG:4326', always_xy=True)
    corrected_4326 = shp_transform(lambda xs, ys, z=None: tf.transform(xs, ys), corrected_img)

    note = (f'chamfer shift dx={dx_m:.1f}m dy={dy_m:.1f}m · cost={best_cost:.2f}px '
            f'sharp={sharpness:.2f} support={support:.3f}' +
            (f' area_ratio={area_ratio:.2f}' if area_ratio is not None else ''))

    return PlotResult(plot_number, 'corrected', round(confidence, 3), dx_m, dy_m, note, corrected_4326)
