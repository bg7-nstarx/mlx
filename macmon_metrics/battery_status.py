"""Local stand-in for genai_services.utils.battery_status."""
import subprocess


def get_battery_status() -> bool:
    """True when the Mac is on AC power (plugged in), False when running on battery."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        return "AC Power" in out
    except Exception:
        return False
