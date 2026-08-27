r"""Video tagger: watch the match and mark events, to build ground truth.

Everything the pipeline produces so far is self-consistent but unvalidated.
This is how that changes: a human marks what actually happened, and every
detector can then be scored against it instead of tuned until its output looks
plausible.

Run:
    .venv\Scripts\python.exe tools\tagger.py [--start MM:SS]

PLAYBACK
    space        play / pause
    a  /  d      step back / forward 1 second
    A  /  D      jump 10 seconds        (shift)
    z  /  c      jump 60 seconds
    ,  /  .      single frame back / forward (paused)
    -  /  =      playback speed down / up (0.25x .. 8x)
    0            reset speed to 1x
    g            go to time (type MM:SS in the terminal)

TAGGING            press once at the moment the event happens
    1  scrum            6  try
    2  lineout          7  penalty / free kick
    3  ruck start       8  stoppage START (injury, reset, long delay)
    4  ruck ball out    9  stoppage END

    5  kick in open play      (play continues)
    t  kick to touch          (play stops, lineout follows)
    p  kick at posts          (conversion or penalty attempt: dead time)

    h  huddle / team talk
    k  kickoff / restart      TAG THIS AFTER EVERY TRY -- the span from a try
                              to the next restart is dead time and would
                              otherwise be credited as attacking territory.

    u            undo last tag
    w            write tags to disk (also autosaves every 20 tags)
    q / esc      quit (prompts if unsaved)

Tags are written to data/derived/tags/<match>_tags.csv and reloaded on restart,
so tagging can be done across several sittings.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("data/video/a-side-vs-msu-2025-09-13.mp4")
OUT_DIR = Path("data/derived/tags")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "a-side-vs-msu-2025-09-13_tags.csv"

VIEW_W, VIEW_H = 1400, 788
BAR_H = 108
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Kick types are separate keys because they mean different things downstream:
# an in-play kick keeps the clock running, a kick to touch ends the passage of
# play, and a kick at posts is pure dead time. Cheap to distinguish while
# watching, impossible to recover afterwards.
EVENTS = {
    ord("1"): ("scrum", (80, 200, 255)),
    ord("2"): ("lineout", (80, 255, 200)),
    ord("3"): ("ruck_start", (60, 220, 60)),
    ord("4"): ("ruck_out", (140, 255, 140)),
    ord("5"): ("kick_in_play", (255, 200, 80)),
    ord("t"): ("kick_to_touch", (255, 160, 40)),
    ord("p"): ("kick_at_posts", (200, 160, 60)),
    ord("6"): ("try", (60, 60, 255)),
    ord("7"): ("penalty", (255, 120, 255)),
    ord("8"): ("stoppage_start", (0, 140, 255)),
    ord("9"): ("stoppage_end", (0, 200, 255)),
    ord("h"): ("huddle", (200, 120, 255)),
    ord("k"): ("kickoff", (255, 255, 120)),
}


def fmt(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        p = [float(v) for v in s.split(":")]
        return p[0] * 60 + p[1]
    return float(s)


class Tagger:
    def __init__(self, start: float):
        self.cap = cv2.VideoCapture(str(VIDEO))
        if not self.cap.isOpened():
            sys.exit(f"cannot open {VIDEO}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dur = self.nframes / self.fps
        self.tags: list[tuple[float, str]] = []
        self.load()
        self.playing = False
        self.speed_i = SPEEDS.index(1.0)
        self.frame = None
        self.dirty = False
        self.msg = ""
        self.seek(start)

    # -- io ---------------------------------------------------------------
    def load(self) -> None:
        if OUT_CSV.exists():
            with OUT_CSV.open() as f:
                self.tags = [(float(r["t"]), r["event"]) for r in csv.DictReader(f)]
            self.tags.sort()
            print(f"loaded {len(self.tags)} existing tags from {OUT_CSV}")

    def save(self) -> None:
        self.tags.sort()
        with OUT_CSV.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "event", "mmss"])
            for t, e in self.tags:
                wr.writerow([f"{t:.3f}", e, fmt(t)])
        self.dirty = False
        self.msg = f"saved {len(self.tags)} tags"
        print(f"  {self.msg} -> {OUT_CSV}")

    # -- navigation -------------------------------------------------------
    @property
    def t(self) -> float:
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps

    def seek(self, t: float) -> None:
        t = float(np.clip(t, 0, self.dur - 0.1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * self.fps))
        self.read()

    def read(self) -> None:
        ok, fr = self.cap.read()
        if ok:
            self.frame = fr
        else:
            self.playing = False

    def step_frames(self, n: int) -> None:
        """Move n frames without a full seek where possible."""
        if n == 1:
            self.read()
        else:
            self.seek(self.t + n / self.fps)

    # -- drawing ----------------------------------------------------------
    def render(self) -> np.ndarray:
        vis = cv2.resize(self.frame, (VIEW_W, VIEW_H))
        canvas = np.zeros((VIEW_H + BAR_H, VIEW_W, 3), np.uint8)
        canvas[:VIEW_H] = vis
        t = self.t

        # header strip
        cv2.rectangle(canvas, (0, 0), (VIEW_W, 34), (0, 0, 0), -1)
        state = "PLAY" if self.playing else "PAUSE"
        cv2.putText(canvas, f"{state}  {fmt(t)} / {fmt(self.dur)}   "
                            f"x{SPEEDS[self.speed_i]:g}   tags {len(self.tags)}"
                            f"{'  *unsaved' if self.dirty else ''}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 255) if self.playing else (200, 200, 200), 2)
        if self.msg:
            cv2.putText(canvas, self.msg, (VIEW_W - 380, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 2)

        # timeline with every tag marked
        y0 = VIEW_H + 14
        cv2.rectangle(canvas, (14, y0), (VIEW_W - 14, y0 + 16), (55, 55, 55), -1)
        span = VIEW_W - 28
        for tt, ev in self.tags:
            px = 14 + int(span * tt / self.dur)
            cv2.line(canvas, (px, y0), (px, y0 + 16), EVENTS_BY_NAME.get(ev, (200, 200, 200)), 1)
        px = 14 + int(span * t / self.dur)
        cv2.line(canvas, (px, y0 - 5), (px, y0 + 21), (255, 255, 255), 2)

        # recent tags, and the key legend
        recent = [f"{fmt(tt)} {ev}" for tt, ev in self.tags[-3:]][::-1]
        cv2.putText(canvas, " | ".join(recent) if recent else "no tags yet",
                    (14, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
        l1 = "1 scrum   2 lineout   3 ruck-start   4 ball-out   6 try   7 penalty"
        l2 = "5 kick-in-play   t kick-to-touch   p kick-at-posts   8/9 stoppage   h huddle   k kickoff/restart"
        cv2.putText(canvas, l1, (14, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)
        cv2.putText(canvas, l2, (14, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)
        return canvas

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        win = "RugbyViz tagger"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        print(__doc__.split("PLAYBACK")[1])
        while True:
            if self.frame is None:
                break
            cv2.imshow(win, self.render())

            if self.playing:
                delay = max(1, int(1000 / (self.fps * SPEEDS[self.speed_i])))
                sp = SPEEDS[self.speed_i]
                if sp > 1:                      # drop frames to go faster
                    for _ in range(int(sp) - 1):
                        self.cap.grab()
                    delay = max(1, int(1000 / self.fps))
                self.read()
            else:
                delay = 20

            k = cv2.waitKey(delay) & 0xFF
            if k == 255:
                continue
            if not self.handle(k):
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def handle(self, k: int) -> bool:
        self.msg = ""
        if k in (ord("q"), 27):
            if self.dirty:
                self.save()
            return False
        if k == ord(" "):
            self.playing = not self.playing
        elif k == ord("a"):
            self.seek(self.t - 1)
        elif k == ord("d"):
            self.seek(self.t + 1)
        elif k == ord("A"):
            self.seek(self.t - 10)
        elif k == ord("D"):
            self.seek(self.t + 10)
        elif k == ord("z"):
            self.seek(self.t - 60)
        elif k == ord("c"):
            self.seek(self.t + 60)
        elif k == ord(","):
            self.seek(self.t - 2 / self.fps)
        elif k == ord("."):
            self.step_frames(1)
        elif k == ord("-"):
            self.speed_i = max(0, self.speed_i - 1)
        elif k == ord("="):
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
        elif k == ord("0"):
            self.speed_i = SPEEDS.index(1.0)
        elif k == ord("g"):
            self.playing = False
            try:
                self.seek(parse_time(input("  go to (MM:SS): ")))
            except Exception as e:
                self.msg = f"bad time: {e}"
        elif k == ord("u"):
            if self.tags:
                t, e = self.tags.pop()
                self.msg = f"undid {e} @ {fmt(t)}"
                self.dirty = True
                print(f"  {self.msg}")
        elif k == ord("w"):
            self.save()
        elif k in EVENTS:
            name, _ = EVENTS[k]
            t = self.t
            self.tags.append((t, name))
            self.dirty = True
            self.msg = f"+ {name} @ {fmt(t)}"
            print(f"  {self.msg}")
            if len(self.tags) % 20 == 0:
                self.save()
        return True


EVENTS_BY_NAME = {name: col for name, col in EVENTS.values()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="0:00")
    args = ap.parse_args()
    Tagger(parse_time(args.start)).run()


if __name__ == "__main__":
    main()
