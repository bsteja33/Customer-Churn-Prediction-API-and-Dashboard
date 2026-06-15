"""Lightweight health monitor — pings /health endpoint and checks model state."""

import os
import sys
import json
import urllib.request
import urllib.error


API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_RETRIES = 3


def _get_memory_mb() -> float:
    """Return current RSS memory usage in MB (cross-platform)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            process = kernel32.GetCurrentProcess()
            info = ctypes.create_string_buffer(72)
            kernel32.GetProcessMemoryInfo(process, info, 72)
            rss = int.from_bytes(info[32:40], "little")
            return rss / (1024 * 1024)
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return 0.0


def check_health() -> dict:
    errors = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(f"{API_URL}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode())
                return body
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            errors.append(str(e))
            if attempt < MAX_RETRIES:
                import time
                time.sleep(1)
    raise RuntimeError(f"Health check failed after {MAX_RETRIES} attempts: {'; '.join(errors)}")


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("health_monitor")
    try:
        data = check_health()
    except RuntimeError as e:
        logger.error("FAIL: %s", e)
        return 1

    status = data.get("status", "unknown")
    model_loaded = data.get("model_loaded", False)
    model_path = data.get("model_path", "unknown")
    mem = _get_memory_mb()

    if status != "healthy":
        logger.error("FAIL: status=%s, model_loaded=%s, memory=%.1fMB", status, model_loaded, mem)
        return 1

    if not model_loaded:
        logger.error("FAIL: model not loaded. path=%s, memory=%.1fMB", model_path, mem)
        return 1

    logger.info("OK: status=%s, model_loaded=%s, path=%s, memory=%.1fMB", status, model_loaded, model_path, mem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
