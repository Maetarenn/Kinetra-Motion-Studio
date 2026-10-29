"""Camera / video-file sources. Every source runs on its own thread and keeps only the newest frame
(no queue backlog, minimum latency)."""
import sys
import threading
import time

import cv2

CAPTURE_PRESETS = {"640x480 @60": (640, 480, 60), "1280x720 @60": (1280, 720, 60),
                   "1280x720 @30": (1280, 720, 30), "1920x1080 @30": (1920, 1080, 30)}


def open_capture(source, preset):
    """source: int camera index or str video path. Returns (cap, info) or (None, '')."""
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        return (cap, f"file {source}") if cap.isOpened() else (None, "")
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(source), backend)
    if not cap.isOpened():
        return None, ""
    cw, ch, cf = CAPTURE_PRESETS.get(preset, (1280, 720, 60))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # raw YUY2 is capped at 10-30 fps on most webcams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
    cap.set(cv2.CAP_PROP_FPS, cf)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, f"{int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps (driver)"


def scan_cameras(max_index=6):
    found = []
    for i in range(max_index):
        cap, _ = open_capture(i, "640x480 @60")
        if cap is not None:
            found.append(i)
            cap.release()
    return found


class CameraSource(threading.Thread):
    def __init__(self, source, preset, name=None):
        super().__init__(daemon=True, name=f"cam-{source}")
        self.source, self.preset = source, preset
        self.label = name or (f"CAM {source}" if not isinstance(source, str) else "VIDEO")
        self.cap, self.info = None, ""
        self.is_file = isinstance(source, str)
        self.running = False
        self.ended = False
        self.fps = 0.0
        self._cond = threading.Condition()
        self._latest, self._seq, self._taken = None, 0, 0
        self.produced = 0

    def open(self):
        self.cap, self.info = open_capture(self.source, self.preset)
        return self.cap is not None

    def run(self):
        self.running = True
        file_fps = (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.is_file else 0.0
        n, t_mark, nxt, fails = 0, time.perf_counter(), time.perf_counter(), 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.ended = True
                    break
                fails += 1
                if fails > 120:
                    self.ended = True
                    break
                time.sleep(0.005)
                continue
            fails = 0
            now = time.perf_counter()
            if self.is_file:                              # play back at the file's own speed
                nxt += 1.0 / file_fps
                if nxt > now:
                    time.sleep(nxt - now)
                now = time.perf_counter()
            n += 1
            if now - t_mark >= 1.0:
                self.fps, n, t_mark = n / (now - t_mark), 0, now
            with self._cond:
                self._latest, self._seq = (frame, now), self._seq + 1
                self.produced += 1
                self._cond.notify_all()
        self.running = False
        with self._cond:
            self._cond.notify_all()

    def get(self, timeout=0.25):
        """Block until a frame newer than the last one returned exists. Returns (frame, t) or None."""
        with self._cond:
            if self._seq == self._taken:
                self._cond.wait(timeout)
            if self._seq == self._taken:
                return None
            self._taken = self._seq
            return self._latest

    def peek(self):
        with self._cond:
            return self._latest

    def stop(self):
        self.running = False
        with self._cond:
            self._cond.notify_all()
        if self.is_alive():
            self.join(2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
