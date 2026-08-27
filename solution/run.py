#!/usr/bin/env python3
"""
Run the chamfer-alignment correction over a village and write predictions.geojson.

    uv run solution/run.py data/34855_vadnerbhairav_chandavad_nashik
    uv run solution/run.py data/34855_vadnerbhairav_chandavad_nashik --only 1145,1403,1476,1710,2647,622
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bhume import load, score, write_predictions
from solution.correct import correct_plot


def run(village_dir: str, only: list[str] | None = None, limit: int | None = None):
    village = load(village_dir)
    plots = village.plots
    if only:
        plots = plots.loc[plots.index.intersection(only)]
    elif limit:
        plots = plots.iloc[:limit]

    rows = []
    t0 = time.time()
    with rasterio.open(village.imagery_path) as isrc, \
         (rasterio.open(village.boundaries_path) if village.boundaries_path else _NullCtx()) as bsrc:
        for i, (pn, row) in enumerate(plots.iterrows()):
            recorded_total = (row.get('recorded_area_sqm') or 0) + (row.get('pot_kharaba_ha') or 0) * 10000
            recorded_total = recorded_total if recorded_total > 0 else None
            res = correct_plot(
                isrc, bsrc, pn, row.geometry,
                recorded_total_sqm=recorded_total,
                map_area_sqm=row.get('map_area_sqm') or 0.0,
            )
            rows.append(res)
            if (i + 1) % 250 == 0:
                dt = time.time() - t0
                print(f'  {i+1}/{len(plots)} plots · {dt:.0f}s elapsed', file=sys.stderr)

    gdf = gpd.GeoDataFrame(
        {
            'plot_number': [r.plot_number for r in rows],
            'status': [r.status for r in rows],
            'confidence': [r.confidence for r in rows],
            'method_note': [r.method_note for r in rows],
            'geometry': [r.geometry for r in rows],
        },
        crs='EPSG:4326',
    )

    n_corr = (gdf['status'] == 'corrected').sum()
    n_flag = (gdf['status'] == 'flagged').sum()
    print(f'{n_corr} corrected, {n_flag} flagged (of {len(gdf)})')

    out = write_predictions(Path(village_dir) / 'predictions.geojson', gdf)
    print(f'wrote {out}')

    if village.example_truths is not None:
        print()
        print(score(gdf, village))

    return gdf


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('village_dir')
    ap.add_argument('--only', type=str, default=None, help='comma-separated plot_numbers')
    ap.add_argument('--limit', type=int, default=None, help='only process the first N plots')
    args = ap.parse_args()
    only = args.only.split(',') if args.only else None
    run(args.village_dir, only=only, limit=args.limit)
