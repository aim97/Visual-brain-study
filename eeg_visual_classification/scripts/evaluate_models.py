"""Latency/FLOPs/parameter-count comparison across models (paper Table 4).
Run directly: python eeg_visual_classification/scripts/evaluate_models.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from thop import profile

import eeg_visual_classification.models as models


def evaluate_model(model, input_size=(1, 128, 440), device="cpu", runs=10):
    """
    Measure evaluation metrics for a given model.
    Returns:
        metrics_dict: Dictionary containing params, macs, flops, and timing statistics in ms.
    """
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # 1. Profile MACs & Params ONCE (Isolated from timing loop to avoid overhead)
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2  # Standard approximation: 1 MAC ≈ 2 FLOPs

    # 2. Warm-up (Crucial for clearing CUDA graph setup/caching overhead)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # 3. Pure Inference Timing Loop (Tracking individual run times)
    run_times = []
    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            # Convert seconds to milliseconds
            run_times.append((end - start) * 1000)

    # Calculate statistics
    metrics_dict = {
        "params": params,
        "macs": macs,
        "flops": flops,
        "time_avg": np.mean(run_times),
        "time_std": np.std(run_times),
        "time_min": np.min(run_times),
        "time_max": np.max(run_times),
    }
    return metrics_dict


def inspect_models(models_dict, input_size=(1, 128, 440), device="cpu"):
    """
    Inspect multiple models and display structural and performance statistics.
    """
    results = []
    print("Profiling models...")
    for name, model in models_dict.items():
        metrics = evaluate_model(model, input_size, device)
        results.append((name, metrics))

    # Sort models by parameter count extracted from thop
    results.sort(key=lambda x: x[1]["params"])

    # Print clean wide table results
    headers = f"{'Model':<25}{'Params':<12}{'MACs':<10}{'FLOPs':<10}{'Avg (ms)':<12}{'Std (ms)':<10}{'Min (ms)':<10}{'Max (ms)':<10}"
    print(f"\n{headers}")
    print("-" * 100)

    for name, m in results:
        # Format large structural numbers (e.g., 1.5M, 2.3G)
        p_str = (
            f"{m['params'] / 1e6:.2f}M" if m["params"] >= 1e6 else f"{m['params']:,.0f}"
        )
        m_str = (
            f"{m['macs'] / 1e6:.2f}M" if m["macs"] < 1e9 else f"{m['macs'] / 1e9:.2f}G"
        )
        f_str = (
            f"{m['flops'] / 1e6:.2f}M"
            if m["flops"] < 1e9
            else f"{m['flops'] / 1e9:.2f}G"
        )

        print(
            f"{name:<25}"
            f"{p_str:<12}"
            f"{m_str:<10}"
            f"{f_str:<10}"
            f"{m['time_avg']:<12.3f}"
            f"{m['time_std']:<10.3f}"
            f"{m['time_min']:<10.3f}"
            f"{m['time_max']:<10.3f}"
        )


if __name__ == "__main__":
    models_to_test = {
        "EEGChannelNet": models.EEGChannelNet(),
        "CWT-SpecCNN": models.SleepingPower(spec_type="cwt", core_model="SimpleEEGCNN"),
        "STFT-SpecCNN": models.SleepingPower(
            spec_type="stft", core_model="SimpleEEGCNN"
        ),
        "CWT_Resent18": models.SleepingPower(spec_type="cwt", core_model="ResNet18"),
        "STFT_ResNet18": models.SleepingPower(spec_type="stft", core_model="ResNet18"),
        "EEGNET": models.EEGNET(),
        "EEGConformer": models.EEGConformer(),
        "ATCNet": models.ATCNet(),
        "ShallowConvNet": models.ShallowConvNet(),
        "DeepConvNet": models.DeepConvNet(),
        "EEGInception": models.EEGInception(),
        "EEGITNet": models.EEGITNet(),
        "BrainDecoder": models.BrainDecoder(),
        "NeuroStream": models.NeuroStream(),
        "NeuroStream4D": models.NeuroStream4D(),
        "proposed": models.NeuroStream4D(bin_size=30, n_blocks=3, base=48),
    }

    inspect_models(
        models_to_test,
        input_size=(16, 1, 128, 440),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
