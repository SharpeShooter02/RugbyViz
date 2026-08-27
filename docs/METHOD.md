# RugbyViz — method and decision log

The reference for how this system works and why it works that way. Consult
before changing anything here; add to it when a decision is made.

Companion documents:
- `docs/findings.md` — experiment results and measurements
- `README.md` — quick start

Last updated 2026-08-27.

---

## 1. Goal

Two statistics from Veo footage of club rugby:

1. **Territory** — where the game was played, in thirds of the pitch
2. **Ruck statistics** — count, location, and ruck speed (formation → ball out)

Territory is delivered. Ruck work has not started.

---

## 2. Architecture

```
per match, once, by hand    mosaic --17 clicks--> pitch metres      [config/]
per frame, automatic        frame  --SIFT-->      anchor --known--> mosaic
                                            compose |
                            frame  ------------------> pitch metres
per frame, model            YOLO11x --> feet px --> pitch metres --> crowd filter
                                                          |
                            territory (arithmetic)  +  rucks (not built)
```

Only the detection step is machine learning. Everything else is classical
geometry, arithmetic, and human input.

**Registration anchors directly, never chains.** Each frame matches against the
temporally nearest anchor frame, whose mosaic homography is already known.
Chaining 11,000 frames would accumulate drift with nothing to correct it, and
would fail silently — territory would degrade over the match with no error
raised.

---

## 3. Decisions

Each entry: what was decided, why, and what evidence supports it.

### D1 — Use YOLO11x off the shelf; do not train a detector
COCO contains ~250,000 labelled humans. Matching that from scratch needs tens
of thousands of labelled rugby frames, which do not exist here. Detection is
also not the bottleneck — see D2. Fine-tuning on ruck frames stays open if
measurement later shows detection failing inside contact.

### D2 — Inference at native 1920, no tiling
Ultralytics defaults to 640, which downscales the frame and misses ~75% of
players. At 1920 essentially every player is found. Tiled inference
*underperformed* full resolution (247 detections vs 475 across 8 frames) —
it fragments players across tile seams. A planned subsystem was deleted.

### D3 — Crowd rejection is geometric, not visual
At 1920 the detector also finds ~40 spectators per frame. Colour cannot
separate them because the crowd wears club colours. Instead every detection's
feet are projected to pitch metres and anything outside the playing area is
discarded. 54% of detections are rejected this way, with no appearance model.

### D4 — Feet, not box centre
The homography maps the ground plane, so the transformed point must be on the
ground. Box centres would place every player metres further away.

### D5 — Asymmetric touchline margins
The two touchlines resolve very differently, so one margin cannot serve both.

| | resolution | margin | evidence |
|---|---|---|---|
| far (y<0) | 1.6 px/m | 2.0 m | crowd forms a dense band from −8 m to −2 m peaking at −5 m, then a cliff: 94 detections in the −6..−4 bin against 9 in −2..0 |
| near (y>70) | 190 px/m | 1.0 m | real players thin out (6, 5, **1** in the 64–70 m bins) then a separate population starts at 70 (11, 22, 4) |
| x | — | 12.0 m | in-goal areas, where play legitimately happens |

Defined once in `tools/rugbyviz/pitch.py` so the tracker and demo cannot drift
apart.

### D6 — Pitch calibration is manual, once per match
The pitch is essentially unmarked, so the standard approach (detect lines, fit
a template) is not viable. The mosaic is static for the whole match, so
mosaic→pitch is solved once by clicking. Two minutes of human effort replaces
a model that could not be trained.

### D7 — Landmarks chosen around what this camera can see
The Veo camera sits on the near touchline near halfway and cannot see the
touchline beneath itself. An early landmark list leaned on near-touchline
intersections that do not exist in the footage, which would have left every
clickable point collinear and the homography unsolvable. The working list uses
the far touchline (y=0), the dashed 5 m and 15 m lines (y=5, 15 — their
crossings with the 22s show as visible "+" marks), goalpost bases (y=32.2,
37.8), and the near 5 m/15 m dashes (y=65, 55).

### D8 — Kit colours are learned, not hard-coded
Hand-tuned HSV thresholds returned 83% "dark" against 17% "red" for two teams
of fifteen: red jerseys in shadow have low saturation and low value, so any
fixed cut-off either swallows them into dark or lets grass through as red, and
one threshold cannot cover a 96-minute afternoon. `fit_kit_colours.py` samples
924 torsos, describes each by chromaticity (channel over total intensity,
which divides out brightness) and clusters. Colour-agnostic, so it works
unchanged for a different strip.

**Kit labels are a hint, never ground truth.** They will be worst in a scrum,
where both jerseys are hidden — exactly the case the cluster work targets.

### D9 — Match periods are human-confirmed, not detected
The rolling-variance detector finds the stationary *core* of half-time but not
its edges (players walking to and from the huddle keep the centroid moving),
and cannot distinguish half-time from a long injury stoppage — at a 3-minute
window the most stationary stretch in this match is an injury, not the break.
A human settles it in a minute. Recorded via `set_periods.py`.

### D10 — Territory is reported under two weightings
Wall-clock credits a three-minute injury at x=20 as three minutes of
territory. Stoppage-excluded does not. Neither is "correct": one answers where
the players were standing, the other where the game was being played, and the
gap measures how much dead time the match contained.

Stoppage detection uses **peak-to-peak range** of the play centroid, not
variance — a scrum drifting a metre and a team jogging back to halfway differ
in range but look similar in variance. Tuned against a known injury stoppage
and a known live stretch: a 60 s window at 12 m catches 100% of the injury
while cutting 4% of live play.

Labelled "excl. stoppages" rather than "ball-in-play" — true ball-in-play is
35–40 min of an 80-minute match because it also excludes every scrum feed,
lineout and restart. That needs event detection.

### D11 — Post-try dead ball is excluded via tags
The 60–90 s after a try was being counted, and counted deep in the *attacking*
22 — so scoring a try inflated your own territory figure. Territory now
excludes `try → next kickoff` and any tagged `stoppage_start → stoppage_end`.
Tag-derived because a tag knows *what* happened; the movement heuristic only
guesses that something stopped. Tested against synthetic tags covering 6.2 min,
of which the heuristic had counted 4.0 min as live.

### D12 — The proper fix for duration weighting is events, not seconds
A scrum's *position* is real territory; the distortion is that a scrum needing
three resets counts three times as much as a quick tap from the same spot.
Excluding stoppages cannot fix that without discarding legitimate territory.
The answer is territory per ruck / per phase / per possession, so a
three-minute injury contributes one event rather than 180 seconds. Blocked on
ruck detection.

---

## 4. Tagging conventions

Defined in `tools/tagger.py`; repeated here because they determine what every
downstream number means.

| event | tag at | end at |
|---|---|---|
| scrum | the 9 **feeds** the ball | ball leaves (8 picks up / 9 clears) |
| lineout | the hooker **throws** | ball won and cleared |
| ruck | contest forms on the ground | ball out (9's hands on it) |
| penalty | the referee's **whistle** | closes any open contest automatically |
| try | grounding | — (dead ball runs to next `kickoff`) |

Set pieces are tagged at the restart rather than the whistle because:
- it separates the **contest** from the **setup**
- setup time still comes free, as the gap between events
- a feed or throw is an unambiguous moment; a knock-on on wide-angle footage
  often is not

Penalties are tagged at the whistle, so dead time runs from the whistle to
whatever restarts play — scrum feed, lineout throw, or kick at posts. This is
consistent, since all three restarts are themselves tagged at the restart.

### Contests ended by a penalty or a try

Press `7` or `6` alone; do not press `e` first. The tagger closes whatever was
open and records the reason — `ruck_end_penalty`, `scrum_end_try` — rather than
a plain `ruck_out`.

This is not cosmetic. **A ruck ended by a penalty never produced ball.** Filing
it as `ruck_out` would enter it into ruck-speed statistics as a ruck that
delivered in N seconds when it delivered nothing, biasing the metric that
matters most.

Kicks deliberately do **not** auto-close: a kick implies the ball was already
out, so the contest ended before it. Press `e` for the ball-out, then the kick.

`8`/`9` stoppage tags are for the **exceptional** only — injuries, long delays,
cards. Routine dead time is implied by the gaps between events.

**Consistency beats correctness of convention.** A systematic offset can be
measured and corrected afterwards; inconsistent marking cannot.

---

## 5. What is validated, and what is not

| | status |
|---|---|
| Camera motion is 8-DOF homography | **measured** — 0.3 px vs 16 px for similarity on a large pan |
| Frame registration | **measured** — 98.3% of samples, median 340 inliers, 0.50 px |
| Calibration self-consistency | **measured** — 0.53 m median residual, 0.14–0.60 m sensitivity to 3 px jitter |
| Detection recall | **eyeballed** on 8 frames, not scored |
| Crowd rejection | **inferred** — the touchline shows as a cliff in the histogram, which is strong but indirect |
| Kit classification | **eyeballed** on one frame (9 red / 11 dark, correct) |
| Player positions in metres | **unvalidated** — self-consistent, no ground truth |
| Territory | **unvalidated** — a systematic 3 m shift of the whole pitch would look identical |
| Ruck detection | not built |

Everything below the line marked *measured* rests on internal consistency. The
tagging pass is what converts it to known error bars.

---

## 6. Open risks

1. **No ground truth yet.** Tagging in progress; 2 tags at time of writing.
2. **Single match, single venue.** Background registration works beautifully
   here because the scene is full of houses and parked cars. A ground with a
   plain hedge behind it may behave very differently. Untested.
3. **Kit labels degrade in contact**, which is where the ruck work needs them.
4. **Tracking is not built**, and rugby breaks off-the-shelf trackers harder
   than any sport — identical kit, total occlusion, 16 bodies in a pile.
5. **`cap.set` seeking** dominates batch runtime and gets slower deeper into
   the file. Sequential decode with frame-skipping would be materially faster
   if higher fps is ever needed for ruck timing.

---

## 7. Files

```
config/<match>.json           calibration, kit model      TRACKED
config/<match>.periods.json   human-confirmed periods     TRACKED
data/video/                   raw footage                 ignored
data/derived/mosaic/          mosaic, homographies        ignored
data/derived/positions/       positions.csv (aggregates)  ignored
                              detections.csv (per player) ignored
data/derived/tags/            human tags                  ignored*
tools/                        pipeline                    TRACKED
```

\* tags represent irreplaceable human effort and should be tracked once the
first pass is complete.

| tool | purpose |
|---|---|
| `build_mosaic.py` | stitch sampled frames into one static pitch mosaic |
| `validate_anchors.py` | cross-check anchor homographies against neighbours |
| `calibrate_pitch.py` | click landmarks → mosaic-to-pitch homography |
| `verify_calibration.py` | draw the pitch back onto the mosaic; bird's-eye |
| `analyse_calibration.py` | resolution and sensitivity by pitch region |
| `fit_kit_colours.py` | learn the two kit colours from the footage |
| `track_positions.py` | batch pass → positions.csv + detections.csv |
| `set_periods.py` | record human-confirmed periods and attacking direction |
| `detect_periods.py` | propose period candidates (does not decide) |
| `territory.py` | territory under both weightings |
| `tagger.py` | video tagger for ground truth |
| `pipeline_demo.py` | end-to-end demo with bird's-eye output |
| `rugbyviz/geometry.py` | from-scratch DLT + RANSAC (teaching, validated vs OpenCV) |
| `rugbyviz/pitch.py` | pitch bounds and the on-pitch test |

---

## 8. Lessons that cost time

Recorded because each was a wrong turn taken more than once.

1. **Look at the image before theorising.** The "impossible player count" was
   diagnosed twice as a bad anchor and once as bad mosaic homographies. Both
   were wrong. Rendering one frame showed it was not match play at all.
2. **RANSAC inlier count is not evidence of a good homography.** 110 inliers
   produced a 1272× area blow-up because they were clustered in one patch.
   Extrapolation, not fitting, is what fails. Hence `validate_homography`.
3. **Check the measurement before believing the measurement.** An apparent
   vanishing-line collapse was a bug in the probe: it clamped its sampling
   interval at the pitch edge without adjusting the divisor.
4. **`.gitignore` has no trailing comments.** `data/video/ # footage` matched
   nothing and committed a 3.4 GB file.
5. **Tune against ground truth, not against plausibility.** The first stoppage
   detector excluded 1% of the match and looked fine.
