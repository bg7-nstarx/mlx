"""
requires-python = ">=3.11"
dependencies = [
    "macmon-python>=0.1.2",
    "ollama>=0.6.1",
    "psutil>=7.2.2",
    "requests>=2.32.5",
]
"""

import subprocess
import json
import shutil
import threading
import time
import os
# import ollama
# import functools
# import re
# import psutil
# import signal
# from statistics import mean

class MacMonTracker:
    def __init__(self, binary_path, interval_ms=500):
        self.binary_path = binary_path
        self.interval_ms = interval_ms
        self.process = None
        self.thread = None
        self.is_running = False
        self.history = []
        self.latest = None

    def _reader(self):
        while self.is_running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                break

            try:
                data = json.loads(line.strip())
                print(data, flush=True)
            except (json.JSONDecodeError, KeyError, ZeroDivisionError):
                continue

    def start(self):
        cmd = [self.binary_path, "pipe", "--interval", str(self.interval_ms), "--soc-info"]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self.is_running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=1)

def find_macmon_binary():
    try:
        from pathlib import Path
        # import macmon
        # package_path = os.path.dirname(macmon.__file__)
        # bin_path = os.path.join(package_path, "bin", "macmon")
        # if os.path.exists(bin_path):
        #     return bin_path
        base_path = Path(__file__).resolve().parent
        if "app.asar.unpacked" in str(base_path):
            macmon_binary = (base_path / ".." / ".." / ".." / ".." / "macmon").resolve()
        else:
            base_path = Path(__file__).resolve().parent
            macmon_binary = (base_path / ".." / ".." / ".." / "macmon").resolve()
        # [mlx-projects] no bundled binary here: use the Homebrew macmon on PATH
        if not macmon_binary.exists():
            return shutil.which("macmon") or "macmon"
        return macmon_binary
    except ImportError:
        pass
    return "macmon"


if __name__ == "__main__":
    macmon_bin = find_macmon_binary()
    tracker = MacMonTracker(macmon_bin, interval_ms=500)

    print("Starting macmon tracking...")
    tracker.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping macmon...")
    finally:
        tracker.stop()
        print("macmon stopped.")
