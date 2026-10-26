"""Local stand-in for genai_services.utils.unified_logger: a named logger that writes to logs/<log_file>."""
import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logger(name: str, source: str = "", log_file: str = "app.log", log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        LOG_DIR.mkdir(exist_ok=True)
        handler = logging.FileHandler(LOG_DIR / log_file)
        handler.setFormatter(logging.Formatter(f"%(asctime)s [{source or name}] %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False
    return logger
