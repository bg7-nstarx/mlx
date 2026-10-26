# MLX Stable Diffusion (from Apple's mlx-examples)

Copied from https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion (MIT license, see LICENSE).

Changes for this project are marked `# [mlx-projects]`:

- `model_io.py`: adds `stabilityai/stable-diffusion-xl-base-1.0` (fp16 weights) and downloads without the saved token;
  `LOCAL_PATHS` maps a repo to a local folder with the same file layout (e.g. `models/stable-diffusion-xl-base-1.0-fp16`),
  which is then read instead of the Hugging Face cache
- `__init__.py`: SDXL size conditioning follows the requested image size (upstream fixes it at 512 x 512 for SDXL-Turbo)
