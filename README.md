# BhuMe boundary correction — submission

## Run it

```bash
uv sync
# drop a village bundle (input.geojson, imagery.tif, boundaries.tif, example_truths.geojson)
# into data/<village_slug>/, then:
uv run solution/run.py data/34855_vadnerbhairav_chandavad_nashik
uv run solution/run.py data/12429_malatavadi_chandgad_kolhapur
```

Writes `predictions.geojson` into the village's own folder, in contract format, and self-scores
it against that village's `example_truths.geojson` if present. `solution/correct.py` has the
actual method, doc-commented inline. `--only 123,456` restricts to specific plot_numbers (useful
for iterating fast); `--limit N` runs just the first N.

`data/<village_slug>/predictions.geojson` for both villages is already committed here, produced
by the code as-is (not hand-edited).

## Method, briefly

For each plot: crop the imagery around it, build an edge map (the `boundaries.tif` hint, backed
up by Canny edges where the hint is thin — tree cover, buildings), distance-transform it, then
search small pixel translations of the plot's own boundary for the one that lands closest to real
edges (chamfer matching). Translation only — no rotation/reshape — since the geometry alone
already showed the drift is dominantly a coherent local offset, and translation-only is harder to
overfit to six example plots.

The search radius is derived from each plot's own footprint size, not a fixed metre value — see
the note below on why.

**Confidence** blends four things, not just match cost: how much the best offset beats a typical/
flat one in the search window (sharpness), how tight the absolute fit is, how much real edge
evidence exists in the patch (thin under trees → discounted), and whether the drawn area agrees
with the recorded 7/12 total (translation can't fix a shape problem, so a bad ratio caps
confidence and can force a flag even when the pixel match looks clean).

## What I found, and what's still open

- A single global per-village shift (the provided baseline) captures most of the drift, but a
  real per-plot residual is left over — the local chamfer search closes most of that gap
  (Vadnerbhairav: median IoU 0.61 official → 0.89 corrected on the 6 example truths).
- **Fixed absolute search radii don't generalize.** A 20m radius tuned by feel on Vadnerbhairav's
  large fields actively hurt Malatavadi — tightly packed small plots let the search jump onto a
  *neighbour's* real edge and confidently land in the wrong place. Fixed it by deriving the radius
  from each plot's own size; this held Vadnerbhairav's accuracy and fixed Malatavadi's calibration
  (Spearman/AUC went from -1.0, i.e. actively anti-correlated, to 1.0 on its 3 example truths).
- **Still open**: Malatavadi's raw accuracy on those 3 truths trails the official start even after
  the fix. Two of the three plots have edges that are genuinely ambiguous with a neighbouring
  plot's edge using only local signal — confidence correctly ranks them as the least trustworthy
  of the three, but doesn't reliably veto them outright on a sample this small, and I stopped
  short of hand-tuning a cutoff to flip exactly these 3 examples, since that's overfitting to the
  public sample rather than a real fix. The real fix I'd try next: use *neighbouring* plots'
  official geometry as mutual-exclusion constraints, so a correction can't cross into a
  neighbour's footprint — structural, not a threshold tweak.
- 3 and 6 example truths are too few to trust calibration numbers as more than directional, per
  the kit's own warning — treat the numbers above as "this direction seems to help", not a score.

## Repo layout

- `bhume/` — provided starter-kit plumbing (I/O, CRS, scoring), unmodified.
- `solution/correct.py` — the actual method (per-plot correction + confidence).
- `solution/run.py` — driver: village → `predictions.geojson`.
- `data/<village_slug>/predictions.geojson` — output for each village attempted.
- `transcripts/` — AI transcripts (see `transcripts/README.md`).
