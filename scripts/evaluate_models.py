import torch
import time
import eeg_visual_classification.models as models


def count_parameters(model):
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, input_size=(1, 128, 440), device="cpu", runs=10):
    """
    Measure evaluation time for a given model.
    Args:
        model: PyTorch model.
        input_size: Shape of the input tensor (batch_size, channels, samples).
        device: 'cpu' or 'cuda'.
        runs: Number of runs to average timing.
    Returns:
        avg_time: Average inference time in seconds.
    """
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warm-up (important for GPU)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / runs
    return avg_time


def inspect_models(models_dict, input_size=(1, 128, 440), device="cpu"):
    """
    Inspect multiple models for parameter count and evaluation time.
    Args:
        models_dict: Dictionary {model_name: model_instance}.
        input_size: Shape of input tensor.
        device: 'cpu' or 'cuda'.
    """
    results = []
    for name, model in models_dict.items():
        params = count_parameters(model)
        time_taken = evaluate_model(model, input_size, device)
        results.append((name, params, time_taken))

    # sort by parameters
    results.sort(key=lambda x: x[1])

    # Print results
    print(f"{'Model':<30}{'Parameters':<20}{'Eval Time (s)':<15}")
    print("-" * 70)
    for name, params, time_taken in results:
        print(f"{name:<30}{params:<20}{time_taken:<15.6f}")


# Example usage:
if __name__ == "__main__":
    models_to_test = {
        "EEGChannelNet": models.EEGChannelNet(),
        "SleepingPowerCWT": models.SleepingPower(
            spec_type="cwt", core_model="SimpleEEGCNN"
        ),
        "SleepingPowerSTFT": models.SleepingPower(
            spec_type="stft", core_model="SimpleEEGCNN"
        ),
        "SleepingPowerResnetCWT": models.SleepingPower(
            spec_type="cwt", core_model="ResNet18"
        ),
        "SleepingPowerResnetSTFT": models.SleepingPower(
            spec_type="stft", core_model="ResNet18"
        ),
        "EEGNET": models.EEGNET(),
        "EEGConformer": models.EEGConformer(),
        "BrainDecoder": models.BrainDecoder(),
        "BrainDecoder3D": models.NeuroStream(),
        "SpectralBrainDecoder3D": models.NeuroStream4D(),
        # "blstm": models.blstm(),
        # "lstm": models.lstm(),
    }

    inspect_models(
        models_to_test,
        input_size=(16, 1, 128, 440),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
