"""Find the live-play periods in the recording.

The file is 96 minutes for an 80-minute match, so it contains time that is not
play. Every time-based stat is wrong if that time is included -- six minutes of
half-time parked at x=74 would register as sustained territory in one team's
half.

Detection signal: during live play the ball, and therefore the players, travel
the length of the pitch. During a break they stop. So the discriminator is not
how many players are visible but whether play is MOVING:

    live  = rolling standard deviation of play_x over a window is high
    break = it collapses toward zero

That is robust to stoppages inside a half (a scrum reset does not move play for
20 s but the window is longer than that) and to substitutes wandering on.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

POSITIONS = Path("data/derived/positions/positions.csv")
OUT = Path("config/a-side-vs-msu-2025-09-13.periods.json")

# A 90 s window was too short: it flagged four "breaks", and rendering frames at
# each showed only one was half-time. The others were a scrum forming, an
# injury stoppage, and a conversion -- all live match time. Rugby stoppages run
# to a couple of minutes, so the window has to be longer than the longest
# stoppage but shorter than half-time.
WINDOW_S = 150.0       # between the longest stoppage and half-time
MOVE_THRESHOLD = 4.0   # metres of rolling std below which play is "not moving"
MIN_PERIOD_S = 600.0   # a half is at least 10 minutes
MIN_BREAK_S = 60.0     # report candidates; a human picks which is half-time


def load():
    rows = list(csv.DictReader(open(POSITIONS)))
    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["play_x"]) if r["play_x"] else np.nan for r in rows])
    n = np.array([int(r["n_on_pitch"]) if r["n_on_pitch"] else 0 for r in rows])
    return t, x, n


def rolling_std(t, x, window):
    """Std of x within +/- window/2 of each sample. NaNs ignored."""
    out = np.full(len(t), np.nan)
    half = window / 2
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half)
    for i in range(len(t)):
        seg = x[lo[i]:hi[i]]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 8:
            out[i] = seg.std()
    return out


def runs(mask, t):
    """Contiguous [start, end] spans where mask is True."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((t[s], t[i - 1])); s = None
    if s is not None:
        out.append((t[s], t[-1]))
    return out


def main() -> None:
    t, x, n = load()
    sd = rolling_std(t, x, WINDOW_S)
    live = np.nan_to_num(sd, nan=0.0) > MOVE_THRESHOLD

    # close short gaps inside a half, then drop short spurious live stretches
    for a, b in runs(~live, t):
        if b - a < MIN_BREAK_S:
            live[(t >= a) & (t <= b)] = True
    for a, b in runs(live, t):
        if b - a < MIN_PERIOD_S:
            live[(t >= a) & (t <= b)] = False

    periods = runs(live, t)
    breaks = [(a, b) for a, b in runs(~live, t) if b - a >= MIN_BREAK_S]

    print(f"rolling std of play_x over {WINDOW_S:.0f}s, "
          f"threshold {MOVE_THRESHOLD:.0f} m\n")
    print("LIVE PLAY PERIODS")
    for i, (a, b) in enumerate(periods, 1):
        m = (t >= a) & (t <= b)
        print(f"  period {i}: {a/60:6.2f} - {b/60:6.2f} min   "
              f"({(b-a)/60:5.2f} min)   play_x {np.nanmin(x[m]):.0f}-{np.nanmax(x[m]):.0f} m"
              f"   median std {np.nanmedian(sd[m]):.1f} m")
    print("\nBREAKS")
    for a, b in breaks:
        m = (t >= a) & (t <= b)
        print(f"  {a/60:6.2f} - {b/60:6.2f} min   ({(b-a)/60:5.2f} min)   "
              f"play parked at x={np.nanmedian(x[m]):.0f} m   "
              f"std {np.nanmedian(sd[m]):.1f} m   {int(np.median(n[m]))} people")

    total_live = sum(b - a for a, b in periods)
    print(f"\ntotal live play: {total_live/60:.1f} min of {t.max()/60:.1f} min recorded")
    print(f"a rugby match is 80 min of play plus stoppages, so 80-95 is expected")

    if len(periods) != 2:
        print(f"\nWARNING: found {len(periods)} periods, expected 2 halves.")
        print("Check the boundaries in the video before trusting them.")

    OUT.write_text(json.dumps({
        "window_s": WINDOW_S, "move_threshold_m": MOVE_THRESHOLD,
        "periods": [{"index": i, "start_s": round(a, 1), "end_s": round(b, 1),
                     "start_min": round(a / 60, 2), "end_min": round(b / 60, 2)}
                    for i, (a, b) in enumerate(periods, 1)],
        "breaks": [{"start_s": round(a, 1), "end_s": round(b, 1)} for a, b in breaks],
        "verified_by_human": False,
    }, indent=2))
    print(f"\nwrote {OUT}")
    print("VERIFY these timestamps in the video before building stats on them.")


if __name__ == "__main__":
    main()
