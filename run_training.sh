#!/bin/bash

# Basic configuration
WANDB_KEY=""
BASE_DIR="logdir/ppo_agent"

# Create base directory
mkdir -p "$BASE_DIR"

# Training configurations
declare -a configs=(
#    "--seed 2345678 --num-envs 64 --rollout-size 32"
#    "--seed 3456789 --num-envs 32 --rollout-size 400"
#    "--seed 456789 --num-envs 1024 --rollout-size 8"
   "--seed 567893 --num-envs 1 --rollout-size 4096"
)

# Array for process IDs
pids=()

# Launch all configurations
for config in "${configs[@]}"; do
    seed=$(echo $config | grep -o 'seed [0-9]*' | cut -d' ' -f2)
    log_dir="$BASE_DIR/seed_${seed}"
    mkdir -p "$log_dir"
    
    echo "Starting training with: $config"
    python3 train.py \
        --wandb-key "$WANDB_KEY" \
        --logdir "$log_dir" \
        $config > "$log_dir/train.log" 2>&1 &
        
    pids+=($!)
done

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All training runs completed"
