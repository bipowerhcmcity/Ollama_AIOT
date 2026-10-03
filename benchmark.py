"""Simple benchmark for Ollama models.

Usage:
    python3 benchmark.py                          # all models in model_files/
    python3 benchmark.py --models qwen_2.5 phi_mini
"""
import argparse
import csv
import re
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
SHOW_URL = "http://localhost:11434/api/show"
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "sample_data_benchmark.csv"
RESULT_FILE = BASE_DIR / "benchmark_results.csv"
N_SAMPLE_BENCHMARK = 5

def default_models():
    # Model names = Modelfile names, e.g. model_files/qwen_2.5.Modelfile -> qwen_2.5
    return sorted(p.stem for p in (BASE_DIR / "model_files").glob("*.Modelfile"))


def load_prompts(n_sample=None):
    prompts = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prompt"):
                prompts.append(row["prompt"])
            else:
                # Send the raw CSV values (same as the curl call). A JSON dict with the
                # long column names makes the model echo the whole JSON back -> ~20x slower.
                prompts.append(",".join(row.values()))
    if n_sample is None:
        return prompts
    else:
        return prompts[:n_sample]


def gpu_percent():
    """GPU utilization: Apple Silicon via ioreg, NVIDIA via nvidia-smi."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            return float(m.group(1))
    except FileNotFoundError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return float(out.strip().splitlines()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return None


class Monitor:
    """Samples CPU / GPU / RAM usage in a background thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.cpu, self.gpu, self.ram = [], [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        psutil.cpu_percent(None)  # first call always returns 0
        while not self._stop.wait(self.interval):
            self.cpu.append(psutil.cpu_percent(None))
            self.ram.append(psutil.virtual_memory().percent)
            g = gpu_percent()
            if g is not None:
                self.gpu.append(g)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def avg(values):
    return round(sum(values) / len(values), 2) if values else None


def ask(model, prompt):
    r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False, "think": False}, timeout=600)
    r.raise_for_status()
    return r.json()


def param_count(model):
    """N = number of model parameters, read from Ollama."""
    r = requests.post(SHOW_URL, json={"model": model}, timeout=60)
    r.raise_for_status()
    return r.json()["model_info"]["general.parameter_count"]


def benchmark(model, prompts):
    print(f"\n=== {model} ===")
    ask(model, "hello")  # warm-up: load the model into memory, not counted
    N = param_count(model)
    print(f"Parameters: {N / 1e9:.2f}B")

    times = []
    p_tokens = g_tokens = 0     # prefill (prompt) / decode (generated) tokens
    t_prefill = t_decode = 0.0  # seconds
    with Monitor() as mon:
        for i, prompt in enumerate(prompts, 1):
            start = time.perf_counter()
            r = ask(model, prompt)
            times.append(time.perf_counter() - start)

            # FLOPs ~= 2 * N * tokens (1 multiply + 1 add per parameter per token)
            p, g = r["prompt_eval_count"], r["eval_count"]
            t_p, t_g = r["prompt_eval_duration"] / 1e9, r["eval_duration"] / 1e9
            p_tokens, g_tokens = p_tokens + p, g_tokens + g
            t_prefill, t_decode = t_prefill + t_p, t_decode + t_g

            print(f"[{i}/{len(prompts)}] {times[-1]:.2f}s | {r['response'].strip()}")
            print(f"    Prefill: {p} token | {2*N*p/1e12:.2f} TFLOPs | {2*N*p/t_p/1e12:.2f} TFLOPS thực tế")
            print(f"    Decode : {g} token | {2*N*g/1e12:.2f} TFLOPs | {g/t_g:.1f} token/s")

    n = len(prompts)
    return {
        "model": model,
        "cpu_percent": avg(mon.cpu),
        "gpu_percent": avg(mon.gpu),
        "ram_percent": avg(mon.ram),
        "avg_time_per_prompt_s": avg(times),
        "avg_tflops_per_prompt": round(2 * N * (p_tokens + g_tokens) / n / 1e12, 3),
        "prefill_tflops_per_s": round(2 * N * p_tokens / t_prefill / 1e12, 2),
        "decode_tflops_per_s": round(2 * N * g_tokens / t_decode / 1e12, 3),
        "decode_tokens_per_s": round(g_tokens / t_decode, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=default_models())
    args = parser.parse_args()

    prompts = load_prompts(N_SAMPLE_BENCHMARK)
    results = [benchmark(m, prompts) for m in args.models]

    print("\n=== Summary ===")
    print(f"{'model':<15}{'CPU %':>10}{'GPU %':>10}{'RAM %':>10}{'Avg s/prompt':>15}"
          f"{'TFLOPs/prompt':>15}{'Prefill TFLOPS':>16}{'Decode TFLOPS':>15}{'token/s':>10}")
    for r in results:
        print(f"{r['model']:<15}{str(r['cpu_percent']):>10}{str(r['gpu_percent']):>10}"
              f"{str(r['ram_percent']):>10}{str(r['avg_time_per_prompt_s']):>15}"
              f"{r['avg_tflops_per_prompt']:>15}{r['prefill_tflops_per_s']:>16}"
              f"{r['decode_tflops_per_s']:>15}{r['decode_tokens_per_s']:>10}")

    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {RESULT_FILE.name}")


if __name__ == "__main__":
    main()
