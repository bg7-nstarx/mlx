"""Runs the TTT models (ttt_vaibhav/ttt_small.py, ttt_medium.py, ttt_large.py) on
prompts/tti_tts_ttt_prompts/ttt_prompts_500w.csv, on Apple Silicon.

    .venv/bin/python run_test.py              # SMALL, MEDIUM and LARGE, each in its own process
    .venv/bin/python run_test.py LARGE        # one size

Results: outputs/ttt/ttt_vaibhav/<SIZE>/Model_Level_Metrics/TTT_<SIZE>_Results.csv (+ System_Level_Metrics/).
"""
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('run_test')


def prepare(path, source):
    """Convert the original checkpoint to a float16 MLX model the first time."""
    if Path(path).exists():
        return
    from huggingface_hub import snapshot_download
    from mlx_lm import convert
    logger.info(f'Converting {source} -> {path} (float16), one-time step...')
    src = snapshot_download(source, token=False, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    convert(hf_path=src, mlx_path=path, dtype='float16')


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')   # every model here is public
    sizes = sys.argv[1:] or ['SMALL', 'MEDIUM', 'LARGE']
    if len(sizes) > 1:   # one fresh process per model, so each starts from a clean GPU state
        for size in sizes:
            subprocess.run([sys.executable, __file__, size], check=True)
        sys.exit()
    from utils.constants import MODEL_PATH
    ttt = importlib.import_module(f'ttt_vaibhav.ttt_{sizes[0].lower()}')   # ttt_small / ttt_medium / ttt_large
    path, source = MODEL_PATH[sizes[0]]
    prepare(path, source)
    out = ttt.process_text_to_text(Path('outputs/ttt/ttt_vaibhav') / sizes[0], Path(path), sizes[0])
    logger.info(f'Done: {out}')
