#!/bin/bash

# ---------------------------
# Base command components
# ---------------------------

BASE_CMD="python dreamer_distill.py"
CONFIGS="--configs defaults metaworld"
TASK="metaworld_drawer_open"
DEVICE="cuda:4"

# ---------------------------
# New hyperparameters
# ---------------------------
USE_VAE="False"
USE_DISTILL="True"
DISTILL_WEIGHT="1.0"
CONDITIONAL_DISTILL="True"

# ---------------------------
# Fixed random seeds
# ---------------------------
SEEDS=(2024)

echo "Using fixed seeds: ${SEEDS[@]}"
echo ""

# ---------------------------
# Run for each teacher type (original_cnn and moe)
# ---------------------------

for SETUP in original moe
do
    if [ "$SETUP" = "original" ]; then
        echo "========================================="
        echo "Running ORIGINAL_CNN teacher setup"
        echo "========================================="

        LOGDIR_BASE="/storage/ssd1/richtsai1103/vid2act/log/metaworld/mt6/10_top50/drawer_open/original_metrics_hi"
        TEACHER_ENCODER_MODE="original_cnn"
        TEACHER_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6_10_top50/original/teacher_model.pt"
        VAE_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6_10_top50/original/vae_model.pt"

    else
        echo "========================================="
        echo "Running MOE teacher setup"
        echo "========================================="

        LOGDIR_BASE="/storage/ssd1/richtsai1103/vid2act/log/metaworld/mt6/10_top50/drawer_open/moe_metrics"
        TEACHER_ENCODER_MODE="moe"
        TEACHER_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6_10_top50/moe/teacher_model.pt"
        VAE_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/mt6_10_top50/moe/vae_model.pt"

    fi

    echo "Logdir base: $LOGDIR_BASE"
    echo "Teacher encoder mode: $TEACHER_ENCODER_MODE"
    echo "Device: $DEVICE"
    echo ""

    # ---------------------------
    # Run the command for each seed
    # ---------------------------
    for seed in "${SEEDS[@]}"
    do
        echo "-----------------------------------------"
        echo "Running $SETUP experiment with seed: $seed"
        echo "-----------------------------------------"

        LOGDIR="${LOGDIR_BASE}_seed${seed}"

        # Run the full command with additional hyperparameters
        $BASE_CMD $CONFIGS \
            --logdir "$LOGDIR" \
            --teacher_encoder_mode $TEACHER_ENCODER_MODE \
            --device $DEVICE \
            --teacher_model_path "$TEACHER_MODEL_PATH" \
            --vae_model_path "$VAE_MODEL_PATH" \
            --task "$TASK" \
            --seed $seed \
            --use_vae $USE_VAE \
            --use_distill $USE_DISTILL \
            --distill_weight $DISTILL_WEIGHT \
            --conditional_distill $CONDITIONAL_DISTILL

        # Check status
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "$SETUP, seed $seed completed successfully"
        else
            echo "$SETUP, seed $seed failed with error code $STATUS"
            # exit 1   # enable if you want to abort on first failure
        fi

        echo ""
    done

    echo "Finished all seeds for setup: $SETUP"
    echo ""
done

echo "All experiments (original_cnn and moe) completed!"
echo "Seeds used: ${SEEDS[@]}"
