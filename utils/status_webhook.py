"""Progress reporting (local stand-in for the production webhook: logs progress instead of calling a webhook)."""
from run_test import logger


def update_progress_and_trigger_webhook(index, total, model_type, **kwargs):
    logger.info(f"{model_type}: {index + 1}/{total} prompts done")
