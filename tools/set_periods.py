"""Record human-confirmed match periods and attacking direction.

Auto-detection cannot do this reliably. The rolling-variance detector finds the
stationary CORE of half-time but not its edges, because players walking to and
from the huddle keep the centroid moving; and it cannot tell half-time from a
long injury stoppage at all -- in this match the most stationary stretch is an
injury, not the break.

A human watching the video settles it in a minute, so that is the input.

Usage (times as MM:SS or seconds):
    python tools/set_periods.py --kickoff 0:00 --half-end 43:00 \
        --second-start 51:30 --full-time 95:00 \
        --team red --first-half-attacks 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("config/a-side-vs-msu-2025-09-13.periods.json")


def parse_time(s: str) -> float:
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kickoff", required=True, help="first-half kickoff")
    ap.add_argument("--half-end", required=True, help="end of first half")
    ap.add_argument("--second-start", required=True, help="second-half kickoff")
    ap.add_argument("--full-time", required=True)
    ap.add_argument("--team", default="red", choices=["red", "dark"],
                    help="which kit is the club's own team")
    ap.add_argument("--first-half-attacks", type=float, default=100.0,
                    choices=[0.0, 100.0],
                    help="pitch x of the try line the team attacks in the first half")
    args = ap.parse_args()

    k, he, ss, ft = (parse_time(args.kickoff), parse_time(args.half_end),
                     parse_time(args.second_start), parse_time(args.full_time))
    if not (k < he < ss < ft):
        raise SystemExit(f"times must increase: {k} < {he} < {ss} < {ft}")

    attacks_1 = args.first_half_attacks
    attacks_2 = 100.0 - attacks_1        # teams swap ends at half-time

    doc = {
        "source": "human-confirmed from video",
        "verified_by_human": True,
        "our_team_kit": args.team,
        "periods": [
            {"index": 1, "name": "first half", "start_s": k, "end_s": he,
             "our_team_attacks_x": attacks_1},
            {"index": 2, "name": "second half", "start_s": ss, "end_s": ft,
             "our_team_attacks_x": attacks_2},
        ],
        "half_time": {"start_s": he, "end_s": ss},
    }
    OUT.write_text(json.dumps(doc, indent=2))

    print(f"first half : {k/60:6.2f} - {he/60:6.2f} min  ({(he-k)/60:5.2f} min)"
          f"   attacking x={attacks_1:.0f}")
    print(f"half-time  : {he/60:6.2f} - {ss/60:6.2f} min  ({(ss-he)/60:5.2f} min)")
    print(f"second half: {ss/60:6.2f} - {ft/60:6.2f} min  ({(ft-ss)/60:5.2f} min)"
          f"   attacking x={attacks_2:.0f}")
    print(f"\nplaying time: {((he-k)+(ft-ss))/60:.1f} min   our team: {args.team}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
