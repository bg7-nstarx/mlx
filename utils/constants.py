"""Model and framework names used by the TTT code in ttt_final_vaibhav/ (local stand-in for the production utils.constants)."""

MODEL_TYPE = {'SMALL': 'TTT_SMALL', 'MEDIUM': 'TTT_MEDIUM', 'LARGE': 'TTT_LARGE'}

MODEL_NAME = {
    'APPLE': {
        'TTT': {
            'SMALL': 'Falcon-H1-1B-Base',
            'MEDIUM': 'Phi-3-mini-128k-instruct',
            'LARGE': 'Qwen3.5-9B',
        },
    },
}

FRAMEWORK = {'APPLE_FRAMEWORK': 'MLX', 'APPLE_FRAMEWORK_TYPE': 'mlx-lm (float16)'}

# Local float16 MLX model folders, and the original checkpoints they are converted from
MODEL_PATH = {
    'SMALL': ('models/Falcon-H1-1B-Base-fp16', 'tiiuae/Falcon-H1-1B-Base'),
    'MEDIUM': ('models/Phi-3-mini-128k-instruct-fp16', 'microsoft/Phi-3-mini-128k-instruct'),
    'LARGE': ('models/Qwen3.5-9B-fp16', 'Qwen/Qwen3.5-9B'),
}
