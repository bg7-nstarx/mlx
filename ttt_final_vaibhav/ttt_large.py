"""TTT MEDIUM — Llama-2-7b-hf (6.7B), float16, MLX on Apple Silicon."""
import os
import gc
import sys
import time
import platform
import pandas as pd
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import snapshot_download
from pathlib import Path

MAX_NEW_TOKENS = 1024   # hard cap: no response exceeds 1,024 output tokens
BASE_STOP = "\n### "    # the base model starting a new section has finished its answer


def load_model(local_path):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This code runs only on Apple Silicon (MLX on the Apple GPU).")
    # Load tokenizer and model (float16 MLX weights) from the local path
    model, tokenizer = load(str(local_path))
    return tokenizer, model


def process_text_to_text(output_folder, model_dir, model_type_key="MEDIUM", **kwargs):
    base_path = Path(__file__).parent.resolve()
    csv_file_path = (base_path / ".." / "prompts/tti_tts_ttt_prompts/ttt_prompts_500w.csv").resolve()
    data = pd.read_csv(csv_file_path)
    if not Path(model_dir).is_dir():
        # Not a local path: treat it as a Hugging Face repo ID and use the copy already in the HF cache
        model_dir = snapshot_download(repo_id=str(model_dir), local_files_only=True)
    model_dir = Path(model_dir)

    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist or is not a directory.")

    results = []
    tokenizer, model = load_model(model_dir)
    sampler = make_sampler(temp=0.0)   # greedy decoding, so results are reproducible

    for index, row in data.iterrows():
        try:
            prompt = row["prompt"]
            text = f"Below is an instruction, followed by a response that completes it.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
            input_ids = tokenizer.encode(text)

            start_time = time.time()
            response = ""
            for r in stream_generate(model, tokenizer, input_ids, max_tokens=MAX_NEW_TOKENS, sampler=sampler):
                response += r.text
                if BASE_STOP in response.lstrip():
                    response = response.lstrip().split(BASE_STOP)[0]
                    break
            end_time = time.time()

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
            })

            print(f"[{row['id']}] {output_tokens} tokens in {total_time:.1f} s ({metrics['tokens_per_second']:.1f} tokens/s)")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            results.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "response": None,
                "total_time": None,
                "time_per_token": None,
                "tokens_per_second": None,
                "error": str(e),
            })

    os.makedirs(output_folder, exist_ok=True)
    output_file_path = Path(output_folder) / f"ttt_results_{model_type_key}.csv"
    pd.DataFrame(results).to_csv(output_file_path, index=False)
    print(f"Saved {len(results)} results to {output_file_path}")

    del tokenizer, model
    gc.collect()
    mx.clear_cache()
    return output_folder


if __name__ == "__main__":
    output_folder = "output"
    model_dir = Path(__file__).parent.resolve() / ".." / "models" / "Qwen3.5-9B-fp16"
    process_text_to_text(output_folder, model_dir, model_type_key="LARGE")