#!/bin/bash

# Directory to save all outputs
OUTDIR="./analysis_out_all_new"
mkdir -p "$OUTDIR"

# Define tasks as "task_name:directory_path"
TASKS=(
    "button_press_topdown:/storage/ssd1/richtsai1103/vid2act/pretrain/metaworld/pretrain_data/top50/button-press-topdown"
    "door_open:/storage/ssd1/richtsai1103/vid2act/pretrain/metaworld/pretrain_data/top50/door-open"
    "drawer_close:/storage/ssd1/richtsai1103/vid2act/pretrain/metaworld/pretrain_data/top50/drawer-close"
    "peg_insert_side:/storage/ssd1/richtsai1103/vid2act/pretrain/metaworld/pretrain_data/top50/peg-insert-side"
    "pick_place:/storage/ssd1/richtsai1103/vid2act/pretrain/metaworld/pretrain_data/top50/pick-place"
    "push:/storage/ssd1/richtsai1103/vid2act/pretrain/metaworld/pretrain_data/top50/push"
)

# Define models as "model_path | encoder_mode | configs | tag"
# Note: Ensure the 'configs' string matches your configs.yaml setup for each model type.
MODELS=(
    "/home/richtsai1103/CRL/COMET/models/metaworld/teacher_moe_new/teacher_model.pt|moe|defaults metaworld metaworld_teacher_moe_pretrain|moe_multihead"
    "/home/richtsai1103/CRL/COMET/models/metaworld/teacher_moe_shared_new/teacher_model.pt|moe|defaults metaworld metaworld_teacher_moe_pretrain|moe_shared"
    "/storage/ssd1/richtsai1103/vid2act/models/mt6_10_top50/original/teacher_model.pt|original_cnn|defaults metaworld|baseline_cnn"
)

# Outer loop: Iterate over the 3 models
for model_info in "${MODELS[@]}"; do
    IFS='|' read -r model_path mode configs model_tag <<< "$model_info"

    echo "==========================================================="
    echo "Starting Analysis for Model: $model_tag"
    echo "Path: $model_path"
    echo "==========================================================="

    # Inner loops: Create unique pairs (6 choose 2 = 15 pairs)
    for ((i=0; i<${#TASKS[@]}; i++)); do
        for ((j=i+1; j<${#TASKS[@]}; j++)); do
            
            # Extract task names and paths
            IFS=':' read -r task_a dir_a <<< "${TASKS[$i]}"
            IFS=':' read -r task_b dir_b <<< "${TASKS[$j]}"

            # Create a unique tag for this specific run
            run_tag="${model_tag}_${task_a}_vs_${task_b}"

            echo ">>> Running pair: $task_a vs $task_b ($model_tag)"

            # 1. Run Gradient Conflict Analysis
            python gradient_conflict.py \
                --configs $configs \
                --teacher_encoder_mode $mode \
                --teacher_model_path "$model_path" \
                --task_a "$task_a" --task_b "$task_b" \
                --task_a_dir "$dir_a" --task_b_dir "$dir_b" \
                --tag "$run_tag" \
                --outdir "$OUTDIR" \
                --device cuda:0

            # 2. Run Latent Projection Analysis (PCA & t-SNE)
            python latent_projection.py \
                --configs $configs \
                --teacher_encoder_mode $mode \
                --teacher_model_path "$model_path" \
                --task_a "$task_a" --task_b "$task_b" \
                --task_a_dir "$dir_a" --task_b_dir "$dir_b" \
                --tag "$run_tag" \
                --outdir "$OUTDIR" \
                --device cuda:0

            # 3. Run Latent Separability Analysis (Silhouette, Probe Acc, Maha)
            python latent_separability.py \
                --configs $configs \
                --teacher_encoder_mode $mode \
                --teacher_model_path "$model_path" \
                --task_a "$task_a" --task_b "$task_b" \
                --task_a_dir "$dir_a" --task_b_dir "$dir_b" \
                --tag "$run_tag" \
                --outdir "$OUTDIR" \
                --device cuda:0 \
                --seeds 0 1 2

        done
    done
done

echo "==========================================================="
echo "All 45 comparisons completed successfully across 3 scripts! Check the '$OUTDIR' folder."
echo "==========================================================="