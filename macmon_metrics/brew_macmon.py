import subprocess
import threading
import json
import time
from datetime import datetime
import logging
import traceback

# [mlx-projects] genai_services isn't part of this project: fall back to the local equivalent.
try:
    from genai_services.utils.battery_status import get_battery_status
except ImportError:
    from macmon_metrics.battery_status import get_battery_status

logger = logging.getLogger(__name__)

class MacmonRunner:
    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        self._stop_flag = threading.Event()
        self._thread = None

    def _run_pipe(self):
        cmd = ["macmon", "pipe", "-i", str(self.interval_ms)]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        try:
            for line in proc.stdout:
                if self._stop_flag.is_set():
                    break

                try:
                    data = json.loads(line)
                    print(data, flush=True)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from macmon: %s", line)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)

    def start(self):
        self._thread = threading.Thread(
            target=self._run_pipe,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join()

_macmon_runner = None

def metrics_monitor_start():
    global _macmon_runner   # [mlx-projects] was missing, so metrics_monitor_stop() never saw the runner

    try:
        _macmon_runner = MacmonRunner(interval_ms=1000)
        _macmon_runner.start()

        print(f"Metrics monitoring started")

    except Exception as e:
        logger.error(
            f"Failed to start metrics monitor: {str(e)}, traceback: {traceback.format_exc()}")


def metrics_monitor_stop():
    global _macmon_runner

    try:
        if _macmon_runner is not None:
            _macmon_runner.stop()
            _macmon_runner = None

        print("Metrics monitoring stopped")

    except Exception as e:
        logger.error(
            f"Failed to stop metrics monitor: {str(e)}, traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    metrics_monitor_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        metrics_monitor_stop()
