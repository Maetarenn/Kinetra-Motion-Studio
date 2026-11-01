"""Process probe for external virtual production clients (Unreal Engine 5 and Blender)."""
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class ClientDetector:
    """Detects whether Unreal Engine 5 or Blender are currently running on the system."""

    def __init__(self, cache_ttl: float = 1.0):
        self.cache_ttl = cache_ttl
        self._last_check = 0.0
        self._cached_status = {
            "ue5_running": False,
            "blender_running": False,
            "ue5_pid": None,
            "blender_pid": None,
        }

    def check(self, force: bool = False) -> dict:
        now = time.time()
        if not force and (now - self._last_check) < self.cache_ttl:
            return self._cached_status

        ue5_running = False
        blender_running = False
        ue5_pid = None
        blender_pid = None

        if not _HAS_PSUTIL:
            return self._cached_status

        try:
            for proc in psutil.process_iter(['pid', 'name']):
                pname = (proc.info.get('name') or '').lower()
                if "unrealeditor" in pname or "unrealengine" in pname:
                    ue5_running = True
                    ue5_pid = proc.info.get('pid')
                elif "blender" in pname:
                    blender_running = True
                    blender_pid = proc.info.get('pid')
                if ue5_running and blender_running:
                    break
        except Exception:
            pass

        self._cached_status = {
            "ue5_running": ue5_running,
            "blender_running": blender_running,
            "ue5_pid": ue5_pid,
            "blender_pid": blender_pid,
        }
        self._last_check = now
        return self._cached_status
