#!/bin/bash

# Base command components

BASE_CMD="python dreamer_distill.py"
CONFIGS="--configs defaults metaworld"
TASK="metaworld_drawer_open"

# this is for original_cnn
# LOGDIR_BASE="/storage/ssd1/richtsai1103/vid2act/log/metaworld/mt6/drawer_open/original"
# TEACHER_ENCODER_MODE="original_cnn" # moe or original_cnn
# DEVICE="cuda:4"
# TEACHER_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6/original/teacher_model.pt"
# VAE_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6/original/vae_model.pt"

# this is for moe
LOGDIR_BASE="/storage/ssd1/richtsai1103/vid2act/log/metaworld/mt6/drawer_open/moe_multihead"
TEACHER_ENCODER_MODE="moe" # moe or original_cnn
DEVICE="cuda:1"
TEACHER_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6/moe_multihead/teacher_model.pt"
VAE_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6/moe_multihead/vae_model.pt"



# Fixed random seeds
SEEDS=(0 123 456 789 2024)

echo "Using fixed seeds: ${SEEDS[@]}"
echo ""

# Run the command for each seed with different logdirs
for seed in "${SEEDS[@]}"
do
    echo "========================================="
    echo "Running experiment with seed: $seed"
    echo "========================================="
    
    # Append seed to logdir to keep results separate
    LOGDIR="${LOGDIR_BASE}_seed${seed}"
    
    # Run the command with proper quoting
    $BASE_CMD $CONFIGS --logdir "$LOGDIR" --teacher_encoder_mode $TEACHER_ENCODER_MODE --device $DEVICE --teacher_model_path "$TEACHER_MODEL_PATH" --vae_model_path "$VAE_MODEL_PATH" --task "$TASK" --seed $seed

    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Seed $seed completed successfully"
    else
        echo "Seed $seed failed with error code $?"
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
    
    echo ""
done

echo "All experiments completed!"
echo "Seeds used: ${SEEDS[@]}"
