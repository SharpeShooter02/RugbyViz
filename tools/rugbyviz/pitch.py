"""Pitch bounds and the on-pitch test. Single source of truth for both the
batch tracker and the demo, so they cannot drift apart.

Margins are asymmetric because the two touchlines resolve very differently in
this camera geometry, and each was chosen from a histogram of detections by
distance beyond the line rather than picked by eye.

FAR touchline (y < 0), margin 2.0 m
    Resolves at ~1.6 px/m. Spectators stand 2-8 m beyond it, peaking at -5 m:

        y -6..-4 : 94 detections
        y -4..-2 : 52
        y -2.. 0 :  9   <- cliff

NEAR touchline (y > 70), margin 1.0 m
    Resolves at ~190 px/m. Substitutes and waiting squads stand just beyond it:

        y 66..68 :  5
        y 68..70 :  1   <- gap: real players have thinned out
        y 70..72 : 11
        y 72..74 : 22   <- separate population

X, margin 12.0 m
    In-goal areas extend past the try lines and play legitimately happens
    there, so this one is deliberately generous.
"""
from __future__ import annotations

import numpy as np

MARGIN_X = 12.0
MARGIN_Y_FAR = 2.0
MARGIN_Y_NEAR = 1.0

# A frame containing more people than this is not live play. Two squads warming
# up put ~50 people inside the bounds, which is a signal rather than a defect.
MAX_LIVE_PLAYERS = 35


def on_pitch(P: np.ndarray, length: float, width: float) -> np.ndarray:
    """Boolean mask of which (x, y) positions in metres are in the playing area."""
    if not len(P):
        return np.zeros(0, dtype=bool)
    return ((P[:, 0] > -MARGIN_X) & (P[:, 0] < length + MARGIN_X) &
            (P[:, 1] > -MARGIN_Y_FAR) & (P[:, 1] < width + MARGIN_Y_NEAR))


def feet_of(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each detection box: where the player meets the ground.

    The homography maps the ground plane, so the transformed point must be on
    it. Using the box centre would place every player metres further away.
    """
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
