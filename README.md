# RugbyViz

Automated match statistics from Veo camera footage for club rugby.

## Goals

1. **Territory** — % of play in each third of the pitch.
2. **Ruck stats** — ruck count, location, and ruck speed (formation -> ball out).

## Approach

The footage available is Veo's 1080p auto-follow crop, not the full panorama.
Both goals are built on one shared intermediate product: **player positions in
real pitch metres, per frame.**

```
per match (manual, ~2 min):  mosaic --click--> pitch coords     [config/]
per frame (automatic):       frame --homography--> mosaic
                                      compose |
                             frame -----------> pitch metres
per frame (model):           player detection --> feet px --> pitch metres
                                                       |
                             territory (arithmetic) + rucks (clustering)
```

Only the detection step is machine learning. Registration and calibration are
classical geometry.

## Key findings so far

- Source video is 1920x1080, 29.97fps, ~96 min. Auto-follow crop only.
- Frame-to-frame camera motion is a **full 8-DOF homography**, not a 3-DOF
  crop-window slide. Similarity transforms fail on large pans (~16px median
  error) where homography holds at ~0.3px. See `docs/findings.md`.
- Registration is highly reliable: 700-1800 SIFT matches per pair, ~95% inliers,
  sub-pixel residuals. The static background (houses, trees, parked cars) does
  the work.
- **The pitch is essentially unmarked** — no 22s, no 5m lines, only a faint
  halfway line and some cones. Standard pitch-template fitting is not viable;
  hence the one-time manual calibration.
- No ground-truth stats exist yet for any match. This is the biggest open risk.

## Documentation

- **`docs/METHOD.md`** — how the system works, every design decision and why,
  tagging conventions, and what is validated versus assumed. Read this first;
  add to it when a decision is made.
- `docs/findings.md` — experiment results and measurements.

## Layout

```
data/video/       raw footage            (gitignored)
data/derived/     frames, mosaics        (gitignored)
config/           calibration per match  (TRACKED - manual work is precious)
tools/            scripts and pipeline
docs/             findings and notes
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Also requires `ffmpeg` on PATH (`winget install Gyan.FFmpeg`).
