"""TTT SMALL — Falcon-H1-1B-Base (1.5B), float16, MLX on Apple Silicon."""
import os
import gc
import sys
import time
import platform
import pandas as pd
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from utils.status_webhook import update_progress_and_trigger_webhook
from genai_services.system_metrics.apple.metrics_monitor import setup_output_directory, metrics_monitor_start, metrics_monitor_stop
from utils.constants import MODEL_TYPE, FRAMEWORK, MODEL_NAME
from pathlib import Path
from run_test import logger

MAX_NEW_TOKENS = 1024   # hard cap: no response exceeds 1,024 output tokens
BASE_STOP = "\n### "    # the base model starting a new section has finished its answer


def load_model(local_path):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This code runs only on Apple Silicon (MLX on the Apple GPU).")
    logger.info(f"Loading model from: {local_path}")
    # Load tokenizer and model (float16 MLX weights) from the local path
    model, tokenizer = load(str(local_path))
    logger.info(f"Using device: {mx.default_device()}  |  weights in memory: {mx.get_active_memory() / 1e9:.2f} GB")
    return tokenizer, model


def process_text_to_text(output_folder, model_dir, model_type_key="SMALL", **kwargs):
    base_path = Path(__file__).parent.resolve()
    csv_file_path = (base_path / ".." / "prompts/tti_tts_ttt_prompts/ttt_prompts_500w.csv").resolve()
    data = pd.read_csv(csv_file_path)
    model_dir = Path(model_dir)

    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist or is not a directory.")

    results = []
    sys_met_paths, model_level_metrics_folder = setup_output_directory(output_folder)
    tokenizer, model = load_model(model_dir)
    sampler = make_sampler(temp=0.0)   # greedy decoding, so results are reproducible

    for index, row in data.iterrows():
        try:
            prompt = row["prompt"]
            # Base model, no chat template: use a plain instruction/response layout
            text = f"Below is an instruction, followed by a response that completes it.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
            input_ids = tokenizer.encode(text)

            processes, npu_process = metrics_monitor_start(index, sys_met_paths, include_npu=True)
            start_time = time.time()
            response = ""
            for r in stream_generate(model, tokenizer, input_ids, max_tokens=MAX_NEW_TOKENS, sampler=sampler):
                response += r.text
                if BASE_STOP in response.lstrip():
                    response = response.lstrip().split(BASE_STOP)[0]
                    break
            end_time = time.time()
            metrics_monitor_stop(processes, npu_process)

            # Metrics over the generated tokens
            total_time = end_time - start_time
            output_tokens = r.generation_tokens
            metrics = {
                "response": response.strip(),
                "time_taken": total_time,
                "tokens_per_second": output_tokens / total_time,
                "time_per_token": total_time / output_tokens,
            }

            results.append({
                "id": row["id"],
                "prompt": prompt,
                "response": metrics["response"],
                "total_time": metrics["time_taken"],
                "time_per_token": metrics["time_per_token"],
                "tokens_per_second": metrics["tokens_per_second"],
                "framework": FRAMEWORK['APPLE_FRAMEWORK'],
                "model": MODEL_NAME['APPLE']['TTT'][model_type_key],
                "framework_type": FRAMEWORK['APPLE_FRAMEWORK_TYPE']
            })
            logger.info(f"[{row['id']}] {output_tokens} tokens in {total_time:.1f} s ({metrics['tokens_per_second']:.1f} tokens/s)")

            update_progress_and_trigger_webhook(index, len(data), MODEL_TYPE[model_type_key], **kwargs)
        except Exception as e:
            logger.error(f"Error on prompt {index}: {e}")

    output_csv_path = os.path.join(model_level_metrics_folder, f"{MODEL_TYPE[model_type_key]}_Results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    del tokenizer, model
    gc.collect()
    mx.clear_cache()
    logger.info(type(results))
    return output_folder
