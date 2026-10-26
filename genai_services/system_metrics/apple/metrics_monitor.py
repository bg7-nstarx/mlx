"""System metrics for each prompt (local stand-in for the production module, built on this project's macmon_metrics).

metrics_monitor_start / metrics_monitor_stop keep the production signatures; macmon samples CPU, GPU, NPU (ANE), power
and temperature every 100 ms and writes <index>_system_metrics.csv into the System_Level_Metrics folder.
"""
from macmon_metrics import metrics_monitor as _mm
from macmon_metrics.metrics_monitor import setup_output_directory  # noqa: F401  (re-exported)


def metrics_monitor_start(index, metrics_folder, include_npu=True):
    _mm.metrics_monitor_start(idx=index, metrics_folder=metrics_folder)   # macmon always includes the NPU
    return None, None


def metrics_monitor_stop(processes=None, npu_process=None):
    _mm.metrics_monitor_stop()
