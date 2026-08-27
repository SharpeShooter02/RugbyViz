r"""Video tagger: watch the match and mark events, to build ground truth.

Everything the pipeline produces so far is self-consistent but unvalidated.
This is how that changes: a human marks what actually happened, and every
detector can then be scored against it instead of tuned until its output looks
plausible.

Run:
    .venv\Scripts\python.exe tools\tagger.py [--start MM:SS] [--width 1600]

PLAYBACK
    space        play / pause
    a  /  d      step back / forward 1 second
    A  /  D      jump 10 seconds        (shift)
    z  /  c      jump 60 seconds
    ,  /  .      single frame back / forward (paused)
    -  /  =      playback speed down / up (0.25x .. 8x)
    0            reset speed to 1x
    g            go to time (type MM:SS in the terminal)

VIEW
    f            toggle fullscreen
    scroll       zoom in / out toward the cursor
    right-drag   pan
    r            reset zoom and pan

TAGGING            press once at the moment the event happens
    1  scrum            6  try
    2  lineout          7  penalty / free kick
    3  ruck start       8  stoppage START
    4  ruck ball out    9  stoppage END

    5  kick in open play      (play continues)
    t  kick to touch          (play stops, lineout follows)
    p  kick at posts          (conversion or penalty attempt: dead time)

    h  huddle / team talk
    k  kickoff / restart      TAG THIS AFTER EVERY TRY -- the span from a try
                              to the next restart is dead time and would
                              otherwise be credited as attacking territory.
    e  END of whatever is open (scrum / lineout / stoppage / huddle)

    u            undo last tag
    w            write tags to disk (also autosaves every 20 tags)
    q / esc      quit (saves if there are unsaved tags)

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

BAR_H = 132
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
    ord("e"): ("end", (170, 170, 170)),
}

OPENING = {"scrum", "lineout", "ruck_start", "stoppage_start", "huddle"}
CLOSERS = {"ruck_start": "ruck_out", "stoppage_start": "stoppage_end"}

# Rows of (key, label) drawn along the bottom. Pairs that open and close
# something are kept adjacent so the relationship is visible at a glance.
LEGEND = [
    [("1", "scrum"), ("e", "end scrum"), ("2", "lineout"), ("e", "end lineout"),
     ("3", "RUCK start"), ("4", "RUCK ball out")],
    [("8", "stoppage start"), ("9", "stoppage end"), ("h", "huddle"), ("e", "end huddle"),
     ("6", "try"), ("k", "kickoff/restart")],
    [("5", "kick in play"), ("t", "kick to touch"), ("p", "kick at posts"),
     ("7", "penalty"), ("u", "undo"), ("w", "save")],
    [("space", "play/pause"), ("a/d", "1s"), ("A/D", "10s"), ("z/c", "60s"),
     ("-/=", "speed"), ("f", "fullscreen"), ("scroll", "zoom"), ("r", "reset view")],
]


def still_open(tags: list[tuple[float, str]]) -> list[str]:
    """Openers with no matching close yet, oldest first.

    'The last opener seen' is not the same as 'the thing currently open': after
    a scrum has been closed and a try tagged, there is nothing left to end, and
    'e' should say so rather than close the scrum a second time.
    """
    stack: list[str] = []
    closes = {v: k for k, v in CLOSERS.items()}
    for _, ev in sorted(tags, key=lambda r: r[0]):
        if ev in OPENING:
            stack.append(ev)
        else:
            want = ev[4:] if ev.startswith("end_") else closes.get(ev)
            if want and want in stack:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] == want:
                        stack.pop(i)
                        break
    return stack


def fmt(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        p = [float(v) for v in s.split(":")]
        return p[0] * 60 + p[1]
    return float(s)


def screen_size() -> tuple[int, int]:
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


class Tagger:
    def __init__(self, start: float, width: int):
        self.cap = cv2.VideoCapture(str(VIDEO))
        if not self.cap.isOpened():
            sys.exit(f"cannot open {VIDEO}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dur = self.nframes / self.fps
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.view_w = width
        self.view_h = int(width * self.src_h / self.src_w)
        self.fullscreen = False

        self.zoom = 1.0                       # 1.0 = whole frame visible
        self.cx, self.cy = self.src_w / 2, self.src_h / 2
        self.dragging = False
        self.drag_from = (0, 0)

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
            self.tags.sort(key=lambda r: r[0])
            print(f"loaded {len(self.tags)} existing tags from {OUT_CSV}")

    def save(self) -> None:
        self.tags.sort(key=lambda r: r[0])
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

    # -- view -------------------------------------------------------------
    def clamp_view(self) -> None:
        self.zoom = float(np.clip(self.zoom, 1.0, 12.0))
        halfw = self.src_w / (2 * self.zoom)
        halfh = self.src_h / (2 * self.zoom)
        self.cx = float(np.clip(self.cx, halfw, self.src_w - halfw))
        self.cy = float(np.clip(self.cy, halfh, self.src_h - halfh))

    def view_to_src(self, px: float, py: float) -> tuple[float, float]:
        halfw = self.src_w / (2 * self.zoom)
        halfh = self.src_h / (2 * self.zoom)
        return (self.cx - halfw + (px / self.view_w) * 2 * halfw,
                self.cy - halfh + (py / self.view_h) * 2 * halfh)

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.drag_from = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx, dy = x - self.drag_from[0], y - self.drag_from[1]
            self.cx -= dx * self.src_w / (self.view_w * self.zoom)
            self.cy -= dy * self.src_h / (self.view_h * self.zoom)
            self.drag_from = (x, y)
            self.clamp_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            before = self.view_to_src(x, y)
            self.zoom *= 1.25 if flags > 0 else 1 / 1.25
            self.clamp_view()
            after = self.view_to_src(x, y)
            self.cx += before[0] - after[0]
            self.cy += before[1] - after[1]
            self.clamp_view()

    def crop(self) -> np.ndarray:
        if self.zoom <= 1.001:
            return self.frame
        halfw = int(self.src_w / (2 * self.zoom))
        halfh = int(self.src_h / (2 * self.zoom))
        x0 = int(np.clip(self.cx - halfw, 0, self.src_w - 2 * halfw))
        y0 = int(np.clip(self.cy - halfh, 0, self.src_h - 2 * halfh))
        return self.frame[y0:y0 + 2 * halfh, x0:x0 + 2 * halfw]

    # -- drawing ----------------------------------------------------------
    def render(self) -> np.ndarray:
        vis = cv2.resize(self.crop(), (self.view_w, self.view_h),
                         interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.view_h + BAR_H, self.view_w, 3), np.uint8)
        canvas[:self.view_h] = vis
        t = self.t
        W = self.view_w

        cv2.rectangle(canvas, (0, 0), (W, 36), (0, 0, 0), -1)
        state = "PLAY " if self.playing else "PAUSE"
        cv2.putText(canvas, f"{state}  {fmt(t)} / {fmt(self.dur)}   "
                            f"x{SPEEDS[self.speed_i]:g}   zoom {self.zoom:.1f}x   "
                            f"tags {len(self.tags)}{'  *unsaved' if self.dirty else ''}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                    (0, 255, 255) if self.playing else (210, 210, 210), 2)

        open_now = still_open(self.tags)
        if open_now:
            cv2.putText(canvas, f"OPEN: {', '.join(open_now)}  -> 'e' ends "
                                f"{open_now[-1]}", (W - 640, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 230, 90), 2)
        if self.msg:
            cv2.putText(canvas, self.msg, (W - 640, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 2)

        # timeline with every tag marked
        y0 = self.view_h + 12
        cv2.rectangle(canvas, (12, y0), (W - 12, y0 + 14), (52, 52, 52), -1)
        span = W - 24
        for tt, ev in self.tags:
            px = 12 + int(span * tt / self.dur)
            cv2.line(canvas, (px, y0), (px, y0 + 14),
                     EVENTS_BY_NAME.get(ev.replace("end_", ""), (190, 190, 190)), 1)
        px = 12 + int(span * t / self.dur)
        cv2.line(canvas, (px, y0 - 4), (px, y0 + 18), (255, 255, 255), 2)

        # key panel
        yy = y0 + 34
        for row in LEGEND:
            xx = 14
            for key, label in row:
                cv2.rectangle(canvas, (xx - 3, yy - 12), (xx + 9 * len(key) + 3, yy + 5),
                              (70, 70, 70), -1)
                cv2.putText(canvas, key, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1)
                xx += 9 * len(key) + 10
                cv2.putText(canvas, label, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (165, 165, 165), 1)
                xx += 8 * len(label) + 22
            yy += 22
        return canvas

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        win = "RugbyViz tagger"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.view_w, self.view_h + BAR_H)
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            if self.frame is None:
                break
            cv2.imshow(win, self.render())

            if self.playing:
                sp = SPEEDS[self.speed_i]
                if sp > 1:
                    for _ in range(int(sp) - 1):
                        self.cap.grab()
                    delay = max(1, int(1000 / self.fps))
                else:
                    delay = max(1, int(1000 / (self.fps * sp)))
                self.read()
            else:
                delay = 20

            k = cv2.waitKey(delay) & 0xFF
            if k == 255:
                continue
            if not self.handle(k, win):
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def handle(self, k: int, win: str) -> bool:
        self.msg = ""
        if k in (ord("q"), 27):
            if self.dirty:
                self.save()
            return False
        if k == ord(" "):
            self.playing = not self.playing
        elif k == ord("f"):
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                sw, sh = screen_size()
                self.view_w = sw
                self.view_h = min(sh - BAR_H, int(sw * self.src_h / self.src_w))
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                self.view_w = 1400
                self.view_h = int(self.view_w * self.src_h / self.src_w)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, self.view_w, self.view_h + BAR_H)
        elif k == ord("r"):
            self.zoom = 1.0
            self.cx, self.cy = self.src_w / 2, self.src_h / 2
            self.msg = "view reset"
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
            self.read()
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
            if name == "end":
                open_now = still_open(self.tags)
                if not open_now:
                    self.msg = "nothing open to end"
                    print(f"  {self.msg}")
                    return True
                name = f"end_{open_now[-1]}"
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
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args()
    print(__doc__.split("PLAYBACK")[1])
    Tagger(parse_time(args.start), args.width).run()


if __name__ == "__main__":
    main()
