"""Generate one text-to-video clip with Lance-3B-Video (lance-mlx) and print its stage timings as JSON.

Runs in the separate .venv-lance environment (lance-mlx pins its own mlx-vlm / mlx-video versions), called by
notebooks/ttv_text_to_video.ipynb once per prompt:
    .venv-lance/bin/python ttv_backends/lance_t2v.py --weights models/Lance-3B-Video-fp16 --prompt "..." --output out.mp4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import mlx.core as mx
import numpy as np
from lance_mlx.pipeline.t2v import TextToVideoPipeline

p = argparse.ArgumentParser()
p.add_argument('--weights', required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--output', required=True)
p.add_argument('--width', type=int, default=480)
p.add_argument('--height', type=int, default=480)
p.add_argument('--frames', type=int, default=33)
p.add_argument('--steps', type=int, default=20)
p.add_argument('--cfg', type=float, default=4.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--fps', type=int, default=12)
args = p.parse_args()

marks = {}

class Marker:
    """Swallow the pipeline's verbose printout, timestamping where denoising and VAE decoding start."""
    def write(self, text):
        now = time.perf_counter()
        if 'step 1/' in text:
            marks.setdefault('denoise', now)
        if 'VAE decode' in text:
            marks.setdefault('decode', now)
        return len(text)

    def flush(self):
        pass

t0 = time.perf_counter()
pipe = TextToVideoPipeline.from_pretrained(Path(args.weights), Path(args.weights) / 'vae.safetensors')
t1 = time.perf_counter()
mx.reset_peak_memory()
sys.stdout = Marker()
frames = pipe.generate(args.prompt, num_frames=args.frames, height=args.height, width=args.width,
                       num_steps=args.steps, cfg_scale=args.cfg, seed=args.seed, verbose=True)
t2 = time.perf_counter()
sys.stdout = sys.__stdout__
iio.imwrite(args.output, np.array(frames), fps=args.fps, codec='libx264', quality=8, macro_block_size=1)
print(json.dumps({
    'model_load (s)': round(t1 - t0, 1),
    'text_encoding (s)': round(marks['denoise'] - t1, 1),   # tokenise + prefill the prompt
    'denoising (s)': round(marks['decode'] - marks['denoise'], 1),
    'vae_decode (s)': round(t2 - marks['decode'], 1),
    'peak_memory (GB)': round(mx.get_peak_memory() / 1e9, 2),
}))
