"""Territory: where was the game played?

Reports the same match under two weightings, because they answer different
questions and can disagree.

  WALL-CLOCK   every sampled second counts equally.
               Simple, and inflated by anything that stops the game while
               leaving players parked somewhere: an injury, a long scrum reset
               sequence, a kicker lining up a conversion. Three minutes of
               nobody playing at x=20 is credited as three minutes of territory.

  EXCL. STOPPAGES  prolonged stationary stretches are excluded.

               This is NOT true ball-in-play. Real ball-in-play in rugby is
               only 35-40 minutes of an 80-minute match, because it also
               excludes every scrum feed, lineout and kick restart. What this
               removes is gross dead time -- injuries, long resets -- which is
               what distorts territory most. Measuring true ball-in-play needs
               event detection.

The honest framing: neither is "correct". Wall-clock answers "where were the
players standing"; ball-in-play answers "where was the game being played". The
gap between them is a measure of how much dead time the match contained.

A scrum is worth dwelling on. Unlike an injury, a scrum's POSITION is real
territory -- a scrum on the opposition 5 m line is a genuinely good place to
be. The distortion is not that it counts, but that a scrum needing three resets
counts three times as much as a quick tap from the same spot. Duration
weighting, not the event itself, is the problem.

The proper fix is to weight by EVENTS rather than seconds -- territory per
ruck, per phase, per possession -- so a three-minute injury contributes one
event, not 180 seconds. That needs the ruck detector, so it is not here yet.
Until then, reporting both weightings at least makes the distortion visible.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

POSITIONS = Path("data/derived/positions/positions.csv")
PERIODS = Path("config/a-side-vs-msu-2025-09-13.periods.json")

# A stationary stretch longer than this is treated as dead time.
# Tuned against a known injury stoppage (67.6-69.8 min, confirmed by rendering
# the frame) and a known stretch of live play (55-60 min):
#
#   window  thr   % match excluded   % injury caught   % live wrongly cut
#       20  2.5              8.0                 18                   9
#       60    8              6.8                 45                   0
#       60   12             16.6                100                   4   <- used
#       60   16             31.9                100                  33
STOPPAGE_MIN_S = 30.0
MOVE_WINDOW_S = 60.0
MOVE_THRESHOLD_M = 12.0    # metres of play movement within the window

ZONES = [("own 22", 0, 22), ("own half", 22, 50),
         ("opp half", 50, 78), ("opp 22", 78, 100)]


def load():
    rows = list(csv.DictReader(open(POSITIONS)))
    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["play_x"]) if r["play_x"] else np.nan for r in rows])
    return t, x


def moving_mask(t, x):
    """True where play is actually travelling.

    Uses the peak-to-peak range of play_x inside a short window rather than a
    standard deviation: a scrum that drifts a metre and a team jogging back to
    halfway look very different in range but similar in variance.
    """
    half = MOVE_WINDOW_S / 2
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half)
    rng = np.full(len(t), np.nan)
    for i in range(len(t)):
        seg = x[lo[i]:hi[i]]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 5:
            rng[i] = seg.max() - seg.min()
    moving = np.nan_to_num(rng, nan=0.0) > MOVE_THRESHOLD_M

    # only call it a stoppage if it persists
    out = moving.copy()
    i = 0
    while i < len(moving):
        if moving[i]:
            i += 1; continue
        j = i
        while j < len(moving) and not moving[j]:
            j += 1
        if t[min(j, len(t) - 1)] - t[i] < STOPPAGE_MIN_S:
            out[i:j] = True          # too short to be dead time
        i = j
    return out


def zones_of(a: np.ndarray) -> dict[str, float]:
    out = {}
    for name, lo, hi in ZONES:
        out[name] = float(((a >= lo) & (a < hi)).mean() * 100) if len(a) else 0.0
    return out


def bar(pct: float, width: int = 40) -> str:
    return "#" * int(round(pct / 100 * width))


def main() -> None:
    t, x = load()
    per = json.loads(PERIODS.read_text())
    if not per.get("verified_by_human"):
        print("WARNING: periods are not human-verified. Run tools/set_periods.py\n")
    moving = moving_mask(t, x)

    print(f"our team: {per['our_team_kit']}   "
          f"zones measured from OUR try line (0) to THEIRS (100)\n")

    agg = {"wall": [], "play": []}
    for p in per["periods"]:
        m = (t >= p["start_s"]) & (t <= p["end_s"]) & ~np.isnan(x)
        # attacking coordinate: 0 = our try line, 100 = theirs
        a = x[m] if p["our_team_attacks_x"] == 100 else 100.0 - x[m]
        mv = moving[m]
        agg["wall"].append(a); agg["play"].append(a[mv])

        dur = (p["end_s"] - p["start_s"]) / 60
        dead = (1 - mv.mean()) * dur
        print(f"{p['name'].upper()}   {p['start_s']/60:.1f}-{p['end_s']/60:.1f} min"
              f"   attacking x={p['our_team_attacks_x']:.0f}")
        print(f"  samples {m.sum()}   dead time {dead:.1f} min of {dur:.1f} "
              f"({100*(1-mv.mean()):.0f}%)")
        wz, pz = zones_of(a), zones_of(a[mv])
        print(f"  {'zone':<10}{'wall-clock':>12}{'excl.stops':>14}   {'':<4}")
        for name, _, _ in ZONES:
            print(f"  {name:<10}{wz[name]:>11.1f}%{pz[name]:>13.1f}%   {bar(pz[name])}")
        print(f"  in opposition half: wall {wz['opp half']+wz['opp 22']:.1f}%   "
              f"in-play {pz['opp half']+pz['opp 22']:.1f}%\n")

    A = np.concatenate(agg["wall"]); B = np.concatenate(agg["play"])
    wz, pz = zones_of(A), zones_of(B)
    print("=" * 60)
    print("FULL MATCH")
    print("=" * 60)
    print(f"  {'zone':<10}{'wall-clock':>12}{'excl.stops':>14}{'diff':>8}")
    for name, _, _ in ZONES:
        print(f"  {name:<10}{wz[name]:>11.1f}%{pz[name]:>13.1f}%"
              f"{pz[name]-wz[name]:>+7.1f}   {bar(pz[name])}")
    print(f"\n  territory (opposition half):")
    print(f"     wall-clock   {wz['opp half']+wz['opp 22']:.1f}%")
    print(f"     excl.stoppages {pz['opp half']+pz['opp 22']:.1f}%")
    excl = 100 * (1 - len(B) / len(A))
    print(f"\n  {excl:.0f}% of sampled time was excluded as stoppages "
          f"({excl/100*86.7:.1f} min).")
    biggest = max(ZONES, key=lambda z: abs(pz[z[0]] - wz[z[0]]))[0]
    print(f"  Largest disagreement is in '{biggest}' "
          f"({pz[biggest]-wz[biggest]:+.1f} points), which is where dead time")
    print(f"  was concentrated. A big gap means stoppages clustered in one part")
    print(f"  of the pitch and the wall-clock number is misleading.")


if __name__ == "__main__":
    main()
