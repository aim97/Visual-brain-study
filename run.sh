#!/bin/bash

# Define your configurations
MODELS=("NeuroStream" "NeuroStream-SST")
DATASETS=("data/processed/semantic_eeg_raw.pth" "data/processed/semantic_eeg_5_95_std.pth" "data/processed/semantic_eeg_14_70_std.pth")
SEEDS=(1 2 3 4 5)

# Ensure the script stops if any individual run crashes critically
set -e

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "======================================================"
            echo "Starting Training: Model=$MODEL | Dataset=$DATASET | Seed=$SEED"
            echo "======================================================"

            # Pass the seed to your python script.
            # Make sure your argparse in lib.py supports '--seed'
            python eeg_semantic_classification.py \
                --model_type "$MODEL" \
                --eeg_dataset "$DATASET" \
                --seed "$SEED" \
                --experiment_name "grid_search_v1"

        done
    done
done

echo "All experiments completed successfully!"
