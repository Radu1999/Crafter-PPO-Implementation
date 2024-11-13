# Crafter RL Agent with PPO

This repository contains an implementation of a Proximal Policy Optimization (PPO) agent for playing the Crafter environment. The agent uses an EfficientNetV2 backbone for processing visual observations and learns to survive and craft items in the game.

## Requirements

```bash
pip install torch torchvision timm wandb crafter
```

## Project Structure

- `train.py`: Main training script
- `agent.py`: PPO agent implementation
- `policy.py`: Neural network policy architecture
- `env.py`: Environment wrapper for Crafter

## Usage

### Basic Training

```bash
python train.py --logdir logs/run_1 --wandb-key YOUR_WANDB_KEY
```

### Important Arguments

- `--logdir`: Directory to save logs and model checkpoints
- `--steps`: Total number of training steps (default: 1,000,000)
- `--eval-interval`: Steps between evaluations (default: 100,000)
- `--eval-episodes`: Number of episodes for evaluation (default: 20)
- `--num-envs`: Number of parallel environments (default: 64)
- `--history-length`: Number of frames to stack (default: 4)
- `--lr`: Learning rate (default: 3e-4)
- `--seed`: Random seed (default: 42)

## Architecture

### Policy Network
The policy network uses a pre-trained EfficientNetV2-S backbone followed by separate policy and value heads. The network processes stacked frames and outputs action probabilities and state values.

Key components:
- EfficientNetV2-S backbone (frozen weights)
- Policy head with 3 fully connected layers
- Value head with 3 fully connected layers

### PPO Agent
The PPO implementation includes:
- Advantage estimation using GAE (Generalized Advantage Estimation)
- Value function loss
- Policy gradient loss with clipping
- Entropy bonus for exploration
- Gradient clipping

## Training Process

1. The agent collects experiences using multiple parallel environments
2. Experiences are stored in a buffer
3. PPO updates are performed using mini-batches
4. Regular evaluation episodes track performance
5. Best model is saved based on evaluation performance
6. Training metrics are logged to Weights & Biases

## Monitoring

The training process logs various metrics to Weights & Biases:
- Episode rewards
- Episode lengths
- Policy loss
- Value loss
- Entropy
- Gradient norms
- Evaluation performance

## Saving and Loading

The best performing model is automatically saved to `{logdir}/best_model.pth`. Evaluation statistics are saved to `{logdir}/eval_stats.pkl`.

## Notes

- The implementation uses CUDA if available, otherwise falls back to CPU
- The EfficientNetV2 backbone weights are frozen by default
- Multiple environments run in parallel for faster data collection
- The policy uses frame stacking for temporal information

## Warning

Make sure your logdir path ends with a number to prevent overwriting previous runs. The code will warn you if:
- The logdir path doesn't end in a number
- The logdir already exists