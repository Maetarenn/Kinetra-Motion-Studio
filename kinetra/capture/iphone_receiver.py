"""iPhone TrueDepth / LiDAR Face MoCap Network Receiver.
Receives real-time 60 FPS 52 ARKit facial blendshapes, head rotations, and eye gaze
directly from Apple Live Link Face, iFacialMocap, and FaceCap iOS apps via UDP."""
import json
import socket
import threading
import time
from typing import Dict, Optional, Tuple
import numpy as np

from ..core.constants import BLENDSHAPE_NAMES


class IPhoneTrueDepthReceiver(threading.Thread):
    """Background UDP listener for iPhone TrueDepth / LiDAR facial capture."""

    DEFAULT_PORT = 49983  # Standard port for iFacialMocap / Live Link Face receiver

    def __init__(self, port: int = DEFAULT_PORT):
        super().__init__(daemon=True, name="iphone-truedepth-receiver")
        self.port = int(port)
        self.running = False
        self._sock: Optional[socket.socket] = None
        self._cond = threading.Condition()

        self._bs_name_to_idx = {name.lower(): idx for idx, name in enumerate(BLENDSHAPE_NAMES)}
        self.latest_blendshapes = np.zeros(52, dtype=np.float32)
        self.latest_head_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
        self.latest_head_translation = np.zeros(3, dtype=np.float32)
        self.device_name: str = "Bekleniyor..."
        self.connected: bool = False
        self.last_packet_time: float = 0.0
        self.packets_received: int = 0
        self.fps: float = 0.0

    def start_listening(self) -> bool:
        if self.running:
            return True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self.port))
            self._sock.settimeout(0.5)
            self.running = True
            self.start()
            return True
        except Exception as exc:
            self.running = False
            if self._sock:
                self._sock.close()
                self._sock = None
            return False

    def stop_listening(self):
        self.running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def run(self):
        fps_count = 0
        fps_timer = time.perf_counter()

        while self.running:
            try:
                data, addr = self._sock.recvfrom(65535)
            except (socket.timeout, OSError):
                if time.time() - self.last_packet_time > 2.0:
                    self.connected = False
                continue

            now = time.time()
            self.last_packet_time = now
            self.connected = True
            self.packets_received += 1
            fps_count += 1

            if time.perf_counter() - fps_timer >= 1.0:
                self.fps = fps_count / (time.perf_counter() - fps_timer)
                fps_count = 0
                fps_timer = time.perf_counter()

            # Parse packet
            self._parse_packet(data, addr)

    def _parse_packet(self, data: bytes, addr: Tuple[str, int]):
        try:
            text = data.decode("utf-8", errors="ignore").strip()
        except Exception:
            return

        # Format A: JSON format
        if text.startswith("{") and text.endswith("}"):
            try:
                d = json.loads(text)
                self.device_name = d.get("device_name", f"iPhone ({addr[0]})")
                bs_data = d.get("blendshapes", {})
                new_bs = np.zeros(52, dtype=np.float32)
                for k, v in bs_data.items():
                    idx = self._bs_name_to_idx.get(k.lower())
                    if idx is not None:
                        new_bs[idx] = float(np.clip(v, 0.0, 1.0))
                with self._cond:
                    self.latest_blendshapes = new_bs
                    if "head_rotation" in d and len(d["head_rotation"]) == 4:
                        self.latest_head_rotation = np.array(d["head_rotation"], dtype=np.float32)
                    self._cond.notify_all()
                return
            except Exception:
                pass

        # Format B: iFacialMocap delimiter format (pipe '|' separated key-value pairs)
        # e.g., "eyeBlinkLeft-0.82|eyeBlinkRight-0.12|...#head-pitch,yaw,roll,x,y,z"
        if "|" in text or "-" in text:
            self.device_name = f"iPhone ({addr[0]})"
            new_bs = np.zeros(52, dtype=np.float32)
            parts = text.split("|")
            for p in parts:
                p = p.strip()
                if "#" in p:
                    # Head segment often attached with #
                    sub = p.split("#")
                    p = sub[0]
                if "-" in p or "=" in p:
                    sep = "-" if "-" in p else "="
                    kv = p.split(sep, 1)
                    if len(kv) == 2:
                        k = kv[0].strip().lower()
                        try:
                            val = float(kv[1].strip())
                            idx = self._bs_name_to_idx.get(k)
                            if idx is not None:
                                new_bs[idx] = float(np.clip(val if val <= 1.0 else val / 100.0, 0.0, 1.0))
                        except ValueError:
                            pass
            with self._cond:
                self.latest_blendshapes = new_bs
                self._cond.notify_all()

    def get_latest(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Returns (blendshapes_52, head_rotation_quat, is_connected)."""
        with self._cond:
            is_active = self.connected and (time.time() - self.last_packet_time < 1.5)
            return self.latest_blendshapes.copy(), self.latest_head_rotation.copy(), is_active
