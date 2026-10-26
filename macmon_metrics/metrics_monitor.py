import atexit
import os
import csv
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# [mlx-projects] genai_services isn't part of this project: fall back to the local equivalents.
try:
    from genai_services.utils.battery_status import get_battery_status
    from genai_services.utils.unified_logger import get_logger
except ImportError:
    from macmon_metrics.battery_status import get_battery_status
    from macmon_metrics.unified_logger import get_logger

logger = get_logger(__name__, source="mac-metrics", log_file="mac_metrics_monitor.log", log_level="DEBUG")

# =========================
# Pydantic Models
# =========================

class CPUTelemetry(BaseModel):
    utilization: Optional[float] = None
    temperature: Optional[float] = None
    power: Optional[float] = None
    ram_gb: Optional[float] = None


class GPUTelemetry(BaseModel):
    utilization: Optional[float] = None
    temperature: Optional[float] = None
    power: Optional[float] = None
    ram_gb: Optional[float] = None


class NPUTelemetry(BaseModel):
    utilization: Optional[float] = None
    power: Optional[float] = None
    ram_gb: Optional[float] = None


class SystemTelemetry(BaseModel):
    total_power: Optional[float] = None


class TelemetrySnapshot(BaseModel):
    timestamp: str
    seconds: float
    isPlugged: bool
    cpu: CPUTelemetry
    gpu: GPUTelemetry
    npu: NPUTelemetry
    system: SystemTelemetry


# =========================
# Utility Functions
# =========================

def bytes_to_gb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


def pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100, 2)

def round2(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None

# =========================
# macmon Tracker
# =========================

class MacMonTracker:
    def __init__(self, binary_path: str, interval_ms: Optional[int] = 100):
        self.binary_path = binary_path
        self.interval_ms = interval_ms
        self.process = None
        self.thread = None
        self.running = False
        self.capture_active = False
        self.capture_start_time: Optional[float] = None
        self.current_snapshots: List[TelemetrySnapshot] = []
        self.capture_is_plugged: bool = False
        self.lock = threading.Lock()

    def _parse_macmon_json(
        self,
        data: Dict[str, Any],
        capture_start_time: float,
        is_plugged: bool,
    ) -> TelemetrySnapshot:
        elapsed_seconds = round(time.time() - capture_start_time, 2)
        return TelemetrySnapshot(
            timestamp=data["timestamp"],
            seconds=elapsed_seconds,
            isPlugged=is_plugged,

            cpu=CPUTelemetry(
                utilization=pct(data.get("pcpu_usage", [None, None])[1]),
                ram_gb=bytes_to_gb(data.get("memory", {}).get("ram_usage")),
                power=round2(data.get("cpu_power")),
                temperature=round2(data.get("temp", {}).get("cpu_temp_avg")),

            ),
            gpu=GPUTelemetry(
                utilization=pct(data.get("gpu_usage", [None, None])[1]),
                ram_gb=None,
                power=round2(data.get("gpu_power")),
                temperature=round2(data.get("temp", {}).get("gpu_temp_avg")),
            ),
            npu=NPUTelemetry(
                utilization=None,
                ram_gb=None,
                power=round2(data.get("ane_power")),
            ),
            system=SystemTelemetry(
                total_power=round2(data.get("sys_power")),
            ),
        )

    def _reader(self):
        while self.running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                break

            try:
                raw = json.loads(line.strip())
                with self.lock:
                    if not self.capture_active or self.capture_start_time is None:
                        continue
                    snapshot = self._parse_macmon_json(
                        raw,
                        self.capture_start_time,
                        self.capture_is_plugged,
                    )
                    self.current_snapshots.append(snapshot)
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    def start(self):
        if self.process and self.process.poll() is None:
            return

        cmd = [
            self.binary_path,
            "pipe",
            "--interval",
            str(self.interval_ms),
            "--soc-info",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def start_capture(self):
        self.start()
        # Query once per capture window to avoid frequent subprocess forks (`pmset` on macOS).
        capture_is_plugged = get_battery_status()
        with self.lock:
            self.capture_start_time = time.time()
            self.current_snapshots = []
            self.capture_is_plugged = capture_is_plugged
            self.capture_active = True

    def stop_capture(self) -> List[TelemetrySnapshot]:
        with self.lock:
            self.capture_active = False
            snapshots = list(self.current_snapshots)
            self.current_snapshots = []
            self.capture_start_time = None
            self.capture_is_plugged = False
        return snapshots

    def stop(self):
        self.running = False
        self.capture_active = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


# =========================
# CSV Writer
# =========================

def write_metrics_to_csv(file_path: str, snapshots: List[TelemetrySnapshot]):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "CPU_Util(%)",
                "CPU_Temperature(C)",
                "cpuPower",
                "Used (GB)",
                "GPU Utilization (%)",
                "GPU_RAM(MB)",
                "GPU_Temperature(C)",
                "gpuPower",
                "npu_usage_percentage",
                "npu_memory_usage_mb",
                "npuPower",
                "total_system_power",
                "seconds",
                "isPlugged",
            ],
        )
        writer.writeheader()

        for s in snapshots:
            writer.writerow({
                "CPU_Util(%)": s.cpu.utilization,
                "CPU_Temperature(C)": s.cpu.temperature,
                "cpuPower": s.cpu.power,
                "Used (GB)": s.cpu.ram_gb,
                "GPU Utilization (%)": s.gpu.utilization,
                "GPU_RAM(MB)": s.gpu.ram_gb,
                "GPU_Temperature(C)": s.gpu.temperature,
                "gpuPower": s.gpu.power,
                "npu_usage_percentage": s.npu.utilization,
                "npu_memory_usage_mb": s.npu.ram_gb,
                "npuPower": s.npu.power,
                "total_system_power": s.system.total_power,
                "seconds": s.seconds,
                "isPlugged": s.isPlugged,
            })


# =========================
# Helper to locate macmon
# =========================

def find_macmon_binary() -> str:
    try:
        import macmon
        pkg_path = os.path.dirname(macmon.__file__)
        bin_path = os.path.join(pkg_path, "bin", "macmon")
        if os.path.exists(bin_path):
            return bin_path
    except ImportError:
        pass
    return "macmon"

def setup_output_directory(output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)
        system_level_metrics_folder = os.path.join(output_folder, "System_Level_Metrics")
        model_level_metrics_folder = os.path.join(output_folder, "Model_Level_Metrics")
        os.makedirs(system_level_metrics_folder, exist_ok=True)
        os.makedirs(model_level_metrics_folder, exist_ok=True)
        return system_level_metrics_folder, model_level_metrics_folder
    except Exception as e:
        raise

# =========================
# Public API (start / stop)
# =========================

_index = None
_metric_folder = None
_tracker = None

def _resolve_macmon_binary() -> Path:
    base_dir_path = Path(__file__).resolve().parent.parent.parent
    logger.info(f"base_dir_path--------->>>>>>, {base_dir_path}")
    if "app.asar.unpacked" in str(base_dir_path):
        macmon_binary = (base_dir_path / ".." / ".." / "macmon").resolve()
        logger.info(f"macmon_binary--------->>>>>>, {macmon_binary}")
    else:
        macmon_binary = (base_dir_path / ".." / "macmon").resolve()
    # [mlx-projects] no bundled binary here: use the Homebrew macmon on PATH
    if not macmon_binary.exists():
        macmon_binary = Path(shutil.which("macmon") or "macmon")
        logger.info(f"bundled macmon not found, using {macmon_binary}")
    return macmon_binary

def _shutdown_tracker():
    global _tracker
    if _tracker:
        _tracker.stop()
        _tracker = None

atexit.register(_shutdown_tracker)

def metrics_monitor_start(idx: int, metrics_folder: str):
    """
    Start collecting macmon metrics for a single iteration.
    """
    global _index, _metric_folder, _tracker

    _index = idx
    _metric_folder = metrics_folder
    if _tracker is None:
        _tracker = MacMonTracker(binary_path=_resolve_macmon_binary())

    _tracker.start_capture()

def metrics_monitor_stop():
    """
    Stop macmon capture and append collected metrics to CSV.
    """
    global _tracker, _index, _metric_folder

    if not _tracker:
        return

    snapshots = _tracker.stop_capture()

    csv_path = os.path.join(
        _metric_folder,
        f"{_index}_system_metrics.csv",
    )

    write_metrics_to_csv(csv_path, snapshots)

    _index = None
    _metric_folder = None

# if __name__ == "__main__":
#     metrics_folder = "./metrics"
#     iterations = 5

#     for idx in range(iterations):
#         print(f"Starting iteration {idx}")

#         metrics_monitor_start(
#             idx=idx,
#             metrics_folder=metrics_folder,
#         )

#         # ---- your model run goes here ----
#         time.sleep(5)
#         # ---------------------------------

#         metrics_monitor_stop()

#         print(f"Finished iteration {idx}\n")
