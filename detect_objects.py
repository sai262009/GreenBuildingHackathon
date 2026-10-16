import argparse
import os
import platform
import ssl
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld, YOLO

if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


CATEGORIES = {
    "Person": (["person"],                                                    0.35, (80, 220, 80)),
    "Mobile": (["cell phone", "smartphone"],                                  0.25, (255, 160, 40)),
    "Laptop": (["laptop"],                                                    0.30, (200, 120, 0)),
    "Light":  (["light bulb", "ceiling light", "tube light", "lamp", "LED light"],
                                                                              0.20, (0, 230, 255)),
    "Light (on)": ([],                                                        0.00, (0, 255, 255)),
    "Fan":    (["fan", "ceiling fan", "table fan", "exhaust fan"],            0.20, (255, 200, 0)),
    "AC":     (["air conditioner", "split air conditioner", "AC unit"],       0.20, (255, 120, 120)),
}


def build_tables():
    prompts, owner = [], []
    for name, (plist, _, _) in CATEGORIES.items():
        for p in plist:
            prompts.append(p)
            owner.append(name)
    return prompts, owner


class Detector(threading.Thread):
    """Runs the model on the newest frame, in the background."""

    def __init__(self, model, owner, imgsz, device, half, min_conf):
        super().__init__(daemon=True)
        self.model, self.owner = model, owner
        self.imgsz, self.device, self.half, self.min_conf = imgsz, device, half, min_conf
        self.lock = threading.Lock()
        self.frame = None
        self.dets = []
        self.infer_fps = 0.0
        self.running = True
        self.interval = 0.0

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def run(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            t = time.time()
            r = self.model.predict(frame, conf=self.min_conf, iou=0.5,
                                   imgsz=self.imgsz, device=self.device,
                                   half=self.half, agnostic_nms=True,
                                   verbose=False)[0]
            dets = [(self.owner[int(b.cls[0])], float(b.conf[0]),
                     tuple(map(int, b.xyxy[0]))) for b in r.boxes]
            self.dets = dets
            time.sleep(0.02)
            self.infer_fps = 0.8 * self.infer_fps + 0.2 / max(time.time() - t, 1e-6)
            time.sleep(max(0.0, self.interval - (time.time() - t)))


def draw(frame, dets, scale):
    counts = {}
    for cat, conf, (x1, y1, x2, y2) in dets:
        _, min_conf, color = CATEGORIES.get(cat, ([], 0.25, (0, 255, 0)))
        if conf < min_conf * scale:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        label = cat if conf >= 1.0 else f"{cat} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(y1 - 5, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return counts


class LightFinder:
    """Finds lights that are switched ON and rejects reflections, white clothes, screens.

    A blob only counts as 'Light (on)' if ALL of these are true:
      1. it is fully saturated white (>=248), compact and small (not a big white area)
      2. it is NOT on/near a person, laptop or phone (shirts, screens, glasses)
      3. it is in the upper part of the frame (ceiling) OR inside a detected 'Light' box
      4. it is not surrounded by other bright stuff (walls, shirts, windows, glare)
      5. it stays in the SAME spot for several frames (reflections move, real lights don't)
    """

    def __init__(self, top_frac=0.55, confirm=10, confirm_in_box=4):
        self.tracks = []          # [cx, cy, hits, miss, box, in_light_box]
        self.top_frac, self.confirm, self.confirm_in_box = top_frac, confirm, confirm_in_box

    def _candidates(self, frame, dets):
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(gray, 248, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        blocked = []
        for cat, _, (x1, y1, x2, y2) in dets:
            if cat in ("Person", "Laptop", "Mobile"):
                px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                blocked.append((x1 - px, y1 - py, x2 + px, y2 + py))
        light_boxes = [b for c, _, b in dets if c == "Light"]

        def inside(cx, cy, boxes):
            return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in boxes)

        out = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(c)
            if not (0.0002 * w * h < area < 0.03 * w * h):
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            if hull == 0 or area / hull < 0.85:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if max(bw, bh) / max(min(bw, bh), 1) > 8:
                continue
            cx, cy = x + bw // 2, y + bh // 2
            if inside(cx, cy, blocked):
                continue
            in_light = inside(cx, cy, light_boxes)
            if not in_light and cy > self.top_frac * h:
                continue
            # surroundings test: ring around the blob must not be mostly bright
            m = max(bw, bh)
            rx1, ry1 = max(x - m, 0), max(y - m, 0)
            rx2, ry2 = min(x + bw + m, w), min(y + bh + m, h)
            ring = gray[ry1:ry2, rx1:rx2]
            rm = np.ones(ring.shape, bool)
            rm[y - ry1:y - ry1 + bh, x - rx1:x - rx1 + bw] = False
            vals = ring[rm]
            if vals.size == 0 or (vals >= 235).mean() > 0.5 or vals.mean() > 200:
                continue
            pad = 6
            box = (max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h))
            out.append((cx, cy, in_light, box))
        return out

    def update(self, frame, dets):
        w = frame.shape[1]
        lim = (0.03 * w) ** 2                     # must stay within ~3% of width
        for t in self.tracks:
            t[3] += 1
        for cx, cy, in_light, box in self._candidates(frame, dets):
            best, bd = None, lim
            for t in self.tracks:
                d = (t[0] - cx) ** 2 + (t[1] - cy) ** 2
                if d < bd:
                    best, bd = t, d
            if best is None:
                self.tracks.append([cx, cy, 1, 0, box, in_light])
            elif best[3] > 0:                     # count once per frame; anchor stays fixed
                best[2] += 1
                best[3], best[4] = 0, box
                best[5] = best[5] or in_light
        self.tracks = [t for t in self.tracks if t[3] <= 8]
        return [("Light (on)", 1.0, t[4]) for t in self.tracks
                if t[3] <= 3 and t[2] >= (self.confirm_in_box if t[5] else self.confirm)]


def set_max_quality(cap, size=None):
    """Ask the webcam for its highest resolution (MJPG is needed for HD/4K at 30 fps)."""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    sizes = [size] if size else [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (640, 480)]
    got = None
    for w, h in sizes:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, f = cap.read()
        if ok and f is not None and f.shape[1] >= w * 0.95 and f.shape[0] >= h * 0.95:
            got = (f.shape[1], f.shape[0])
            break
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    if got is None:
        ok, f = cap.read()
        if ok and f is not None:
            got = (f.shape[1], f.shape[0])
    if got:
        print(f"Camera running at {got[0]}x{got[1]}")


def hud(frame, counts, cam_fps, ai_fps, scale, flip):
    lines = [f"Video {cam_fps:.0f} fps | AI {ai_fps:.1f} fps | sens x{scale:.1f}",
             f"Flip: {'ON' if flip else 'OFF'}  (press M to toggle)"]
    lines += [f"{k}: {v}" for k, v in sorted(counts.items())]
    h = 24 * len(lines) + 10
    ov = frame.copy()
    cv2.rectangle(ov, (6, 6), (320, 6 + h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (14, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


class LatestFrame:
    """Reads a live camera (webcam index, DroidCam/Iriun, or http stream) in a background
    thread and always returns the NEWEST frame -> no growing delay, display never blocks."""

    def __init__(self, src, backend=0, size=None):
        self.src, self.backend, self.size = src, backend, size
        self.frame, self.new, self.running = None, False, True
        self.lock = threading.Lock()
        self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        if isinstance(self.src, int) and platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        for be in backends:
            self.cap = cv2.VideoCapture(self.src, be)
            if self.cap.isOpened():
                break
            self.cap.release()
        if isinstance(self.src, int):
            set_max_quality(self.cap, self.size)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def isOpened(self):
        return self.cap.isOpened()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                print("Camera/stream lost, reconnecting...")
                time.sleep(1)
                if not self.running:
                    break
                self.cap.release()
                self._open()
                continue
            with self.lock:
                self.frame, self.new = f, True

    def read(self):
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.lock:
                if self.new:
                    self.new = False
                    return True, self.frame.copy()
            time.sleep(0.002)
        return False, None

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


class MjpegStream:
    """Fallback for phone apps (DroidCam etc.) when OpenCV's own stream reader fails.
    Reads the MJPEG stream directly over HTTP and decodes the newest JPEG frame."""

    def __init__(self, url):
        self.url, self.frame, self.new, self.running = url, None, False, True
        self.lock, self._first = threading.Lock(), threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self.ok = self._first.wait(8)

    def isOpened(self):
        return self.ok

    def _loop(self):
        while self.running:
            try:
                resp = urllib.request.urlopen(self.url, timeout=5,
                                              context=ssl._create_unverified_context())
                buf = b""
                while self.running:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    buf += chunk
                    last = None
                    while True:
                        a = buf.find(b"\xff\xd8")
                        b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1
                        if a == -1 or b == -1:
                            break
                        last, buf = buf[a:b + 2], buf[b + 2:]
                    if last is not None:
                        f = cv2.imdecode(np.frombuffer(last, np.uint8), cv2.IMREAD_COLOR)
                        if f is not None:
                            with self.lock:
                                self.frame, self.new = f, True
                            self._first.set()
            except Exception as e:
                if self.running:
                    print(f"Stream error ({e}), retrying...")
                    time.sleep(1)

    def read(self):
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.lock:
                if self.new:
                    self.new = False
                    return True, self.frame.copy()
            time.sleep(0.002)
        return False, None

    def release(self):
        self.running = False


class ScreenGrab:
    """Live-captures a region of your SCREEN (e.g. the DroidCam client's video preview).
    Needs no virtual camera, no driver and no new install (uses Pillow, which ultralytics
    already installed)."""

    def __init__(self):
        from PIL import ImageGrab
        self.grab = ImageGrab.grab
        self.ok, self.frame, self.new, self.running = False, None, False, True
        self.lock = threading.Lock()
        print("\nShow the DroidCam client with the phone video visible on screen.")
        print("Capturing your screen in 4 seconds - switch to the DroidCam window now...")
        for i in range(4, 0, -1):
            print(f"  {i}", flush=True)
            time.sleep(1)
        full = cv2.cvtColor(np.array(self.grab(all_screens=True)), cv2.COLOR_RGB2BGR)
        self.sw, self.sh = full.shape[1], full.shape[0]
        k = min(1.0, 1400 / self.sw, 800 / self.sh)
        small = cv2.resize(full, None, fx=k, fy=k) if k < 1 else full
        r = cv2.selectROI("Drag a box around ONLY the phone video, then press ENTER", small, False)
        cv2.destroyAllWindows()
        x, y, w, h = [int(v / k) for v in r]
        self.bbox = (x, y, x + w, y + h)
        self.ok = w > 20 and h > 20
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def isOpened(self):
        return self.ok

    def _loop(self):
        while self.running:
            img = self.grab(bbox=self.bbox, all_screens=True)
            f = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            with self.lock:
                self.frame, self.new = f, True
            time.sleep(0.01)

    def read(self):
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.lock:
                if self.new:
                    self.new = False
                    return True, self.frame.copy()
            time.sleep(0.002)
        return False, None

    def place_window(self, win):
        """Put our window on the other half of the screen so it never captures itself."""
        half = max(self.sw // 2 - 30, 320)
        cv2.resizeWindow(win, half, int(half * 0.62))
        left = (self.bbox[0] + self.bbox[2]) / 2 < self.sw / 2
        cv2.moveWindow(win, self.sw // 2 + 10 if left else 10, 30)

    def release(self):
        self.running = False


class WindowGrab:
    """Finds the DroidCam client window BY ITS TITLE and captures just that window.
    No dragging a box, no countdown. It also moves the DroidCam window to the top-left and
    puts the detection window next to it, so they never overlap. (Windows only.)"""

    SKIP_CLASSES = {"CabinetWClass", "Chrome_WidgetWin_1", "MozillaWindowClass",
                    "ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS"}

    def __init__(self, title_part="droidcam", crop=(0, 0, 0, 0)):
        self.ok, self.frame, self.new, self.running = False, None, False, True
        self.lock, self.crop, self.hwnd = threading.Lock(), crop, None
        if platform.system() != "Windows":
            print("Window capture only works on Windows.")
            return
        import ctypes
        from ctypes import wintypes
        from PIL import ImageGrab
        self.ct, self.wt, self.grab = ctypes, wintypes, ImageGrab.grab
        self.u = ctypes.windll.user32
        self.title = title_part.lower()
        self.hwnd = self._find()
        if not self.hwnd:
            print(f"Couldn't find a window with '{title_part}' in its title.")
            print("Open the DroidCam client first and make sure its window is open.")
            return
        r = wintypes.RECT()
        self.u.ShowWindow(self.hwnd, 9)
        self.u.GetWindowRect(self.hwnd, self.ct.byref(r))
        self.win_w, self.win_h = r.right - r.left, r.bottom - r.top
        self.u.MoveWindow(self.hwnd, 0, 0, self.win_w, self.win_h, True)
        self.u.SetForegroundWindow(self.hwnd)
        time.sleep(0.5)
        print(f"Found DroidCam window ({self.win_w}x{self.win_h}). Keep it visible, don't cover it.")
        self.ok = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _find(self):
        ct, wt, u = self.ct, self.wt, self.u
        found = []

        def cb(hwnd, _):
            if u.IsWindowVisible(hwnd):
                n = u.GetWindowTextLengthW(hwnd)
                if n:
                    buf = ct.create_unicode_buffer(n + 1)
                    u.GetWindowTextW(hwnd, buf, n + 1)
                    cls = ct.create_unicode_buffer(256)
                    u.GetClassNameW(hwnd, cls, 256)
                    t = buf.value.lower()
                    if (self.title in t and cls.value not in self.SKIP_CLASSES
                            and "live detection" not in t and "visual studio" not in t):
                        r = wt.RECT()
                        u.GetWindowRect(hwnd, ct.byref(r))
                        found.append(((r.right - r.left) * (r.bottom - r.top), hwnd))
            return True

        proc = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        u.EnumWindows(proc(cb), 0)
        return max(found)[1] if found else None

    def _bbox(self):
        r, p = self.wt.RECT(), self.wt.POINT(0, 0)
        self.u.GetClientRect(self.hwnd, self.ct.byref(r))
        self.u.ClientToScreen(self.hwnd, self.ct.byref(p))
        l, t, rr, b = self.crop
        return (p.x + l, p.y + t, p.x + r.right - rr, p.y + r.bottom - b)

    def isOpened(self):
        return self.ok

    def _loop(self):
        while self.running:
            try:
                if self.u.IsIconic(self.hwnd) or not self.u.IsWindow(self.hwnd):
                    time.sleep(0.3)
                    continue
                x1, y1, x2, y2 = self._bbox()
                if x2 - x1 < 20 or y2 - y1 < 20:
                    time.sleep(0.1)
                    continue
                img = self.grab(bbox=(x1, y1, x2, y2), all_screens=True)
                f = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                with self.lock:
                    self.frame, self.new = f, True
            except Exception as e:
                print("Capture error:", e)
                time.sleep(0.5)
            time.sleep(0.01)

    def read(self):
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.lock:
                if self.new:
                    self.new = False
                    return True, self.frame.copy()
            time.sleep(0.002)
        return False, None

    def place_window(self, win):
        """Detection window goes right next to the DroidCam window (no overlap)."""
        sw = self.u.GetSystemMetrics(0)
        x = self.win_w + 20
        w = max(sw - x - 10, 560)
        cv2.resizeWindow(win, w, int(w * 0.62))
        cv2.moveWindow(win, x, 0)

    def release(self):
        self.running = False


def open_camera(src, crop=(0, 0, 0, 0), cam_size=None):
    if isinstance(src, str) and (src.lower() == "droidcam" or src.lower().startswith("window:")):
        return WindowGrab("droidcam" if src.lower() == "droidcam" else src[7:], crop)
    if isinstance(src, str) and src.lower() == "screen":
        return ScreenGrab()
    if isinstance(src, int):
        return LatestFrame(src, cv2.CAP_DSHOW if platform.system() == "Windows" else 0, cam_size)
    if "://" in src:
        cam = LatestFrame(src)
        time.sleep(1.5)
        if cam.isOpened():
            return cam
        cam.release()
        print("OpenCV couldn't open the stream, trying direct MJPEG reader...")
        return MjpegStream(src)
    return cv2.VideoCapture(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="0 = laptop webcam | droidcam = capture the DroidCam client window | http://IP:PORT/video | video file")
    ap.add_argument("--model", default="yolov8s-worldv2.pt",
                    help="faster: yolov8s-worldv2.pt (default) | more accurate: yolov8l-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=416, help="AI input size, lower = faster (320-640)")
    ap.add_argument("--width", type=int, default=1920,
                    help="max width of the displayed video (default 1920; lower = faster)")
    ap.add_argument("--det-width", type=int, default=640,
                    help="width of the copy the AI looks at (lower = faster, higher = finds small things)")
    ap.add_argument("--cam-size", default="",
                    help="force camera resolution, e.g. 1280x720. Default: highest the camera supports")
    ap.add_argument("--ai-fps", type=float, default=4,
                    help="max AI updates per second (keeps CPU free so video stays smooth)")
    ap.add_argument("--custom", action="store_true", help="--model is your own trained YOLO model")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--crop", default="0,0,0,0",
                    help="trim the DroidCam window, pixels: LEFT,TOP,RIGHT,BOTTOM (e.g. 0,0,0,80)")
    ap.add_argument("--flip", type=int, default=1, choices=[0, 1],
                    help="1 = flip image horizontally to undo mirroring (default), 0 = leave as is")
    ap.add_argument("--list-cameras", action="store_true",
                    help="show which camera numbers work, then exit")
    args = ap.parse_args()

    if args.list_cameras:
        print("Scanning cameras (close Zoom/Teams/Camera app first)...")
        found = False
        for i in range(6):
            for name, be in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)):
                c = cv2.VideoCapture(i, be)
                ok, f = c.read() if c.isOpened() else (False, None)
                c.release()
                if ok:
                    print(f"  camera {i} works  ({f.shape[1]}x{f.shape[0]}, backend {name})")
                    found = True
                    break
        if not found:
            print("  No cameras found. DroidCam virtual camera is not installed/running.")
        return

    if torch.cuda.is_available():
        device, half = 0, True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, half = "mps", False
    else:
        device, half = "cpu", False
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print(f"Device: {device}")

    print("Loading model (first run downloads files, please wait)...")
    prompts, owner = build_tables()
    if args.custom:
        model = YOLO(args.model)
        owner = [n.capitalize() for n in model.names.values()]
    else:
        model = YOLOWorld(args.model)
        model.set_classes(prompts)
    print("Model ready. Opening camera...")

    src = int(args.source) if args.source.isdigit() else args.source
    cam_size = tuple(int(v) for v in args.cam_size.lower().split("x")) if args.cam_size else None
    cap = open_camera(src, tuple(int(v) for v in args.crop.split(",")), cam_size)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/source: {args.source}\n"
                 "Run:  python detect_objects.py --list-cameras   to see which numbers work.")

    min_conf = min(c[1] for c in CATEGORIES.values() if c[0]) * 0.5
    det = Detector(model, owner, args.imgsz, device, half, min_conf)
    det.interval = 1.0 / args.ai_fps
    det.start()

    win = "Live Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1000, 600)
    if isinstance(cap, (ScreenGrab, WindowGrab)):
        cap.place_window(win)
    fullscreen = args.fullscreen
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    flip = bool(args.flip)
    lights = LightFinder()
    scale, cam_fps, t0 = 1.0, 30.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)           # undo mirror image
        if frame.shape[1] > args.width:
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * args.width / frame.shape[1])),
                               interpolation=cv2.INTER_AREA)
        fw = frame.shape[1]
        if fw > args.det_width:      # AI + light finder work on a smaller copy, display stays sharp
            small = cv2.resize(frame, (args.det_width, int(frame.shape[0] * args.det_width / fw)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ratio = fw / small.shape[1]
        det.submit(small.copy())
        dets_small = list(det.dets)
        lit = lights.update(small, dets_small)
        dets = [(c, cf, tuple(int(v * ratio) for v in b)) for c, cf, b in dets_small + lit]
        counts = draw(frame, dets, scale)

        now = time.time()
        cam_fps = 0.9 * cam_fps + 0.1 / max(now - t0, 1e-6)
        t0 = now
        hud(frame, counts, cam_fps, det.infer_fps, scale, flip)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif k == ord("m"):
            flip = not flip
        elif k in (ord("+"), ord("=")):
            scale = min(scale + 0.1, 3.0)
        elif k in (ord("-"), ord("_")):
            scale = max(scale - 0.1, 0.3)

    det.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()