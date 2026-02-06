# Snake AI - Reinforcement Learning

A deep reinforcement learning project that trains an AI agent to play the classic Snake game using Proximal Policy Optimization (PPO) from Stable Baselines3.

## Requirements

> **IMPORTANT:** This project requires **Python 3.11**. Other Python versions are not supported and may cause compatibility issues with the dependencies.

## Features

- Classic Snake game implementation with Pygame
- Multiple training environments with different complexity levels
- Configurable neural network architectures
- Run multiple experiments in parallel with automatic comparison
- Real-time visualization of trained AI playing
- TensorBoard integration for training monitoring
- Pre-trained model included

## Project Structure

```
snake_ai/
├── snake_game.py                # Base Snake game with Pygame rendering
├── snake_env.py                 # Advanced environment (curriculum learning)
├── snake_env_simple.py          # Simple environment (777-dim observations)
├── snake_env_simple_minimal.py  # Minimal environment (20-dim observations)
├── train_ai.py                  # Training script for advanced environment
├── train_simple_ai.py           # Training script for simple environment
├── train_minimal_ai.py          # Configurable training (arch, steps, device)
├── train_multiple_minimal_ai.py # Run multiple experiments in parallel
├── watch_ai_play.py             # Visualize trained AI (advanced)
├── watch_simple_ai.py           # Visualize trained AI (simple)
├── watch_minimal_ai.py          # Visualize & compare models (minimal)
├── models/                      # Saved trained models
│   └── *.zip                    # Trained model files
├── logs/                        # Training logs (gitignored)
│   ├── simple/                  # Simple environment logs
│   ├── minimal_*/               # Minimal environment logs per experiment
│   ├── experiments/             # Experiment output logs
│   └── tensorboard/             # TensorBoard logs for all experiments
├── requirements.txt             # Python dependencies
└── test_*.py                    # Various test scripts
```

## Installation

1. **Ensure you have Python 3.11 installed:**
   ```bash
   python3.11 --version
   ```

2. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd snake_ai
   ```

3. **Create and activate a virtual environment (recommended):**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Play the Game Yourself

```bash
python snake_game.py
```

Use arrow keys to control the snake. Press `R` to restart after game over.

### Watch Pre-trained AI Play

```bash
# Watch default model
python watch_minimal_ai.py

# Watch a specific model
python watch_minimal_ai.py --model tiny_10x10

# List all available models
python watch_minimal_ai.py --list

# Test performance (no graphics)
python watch_minimal_ai.py --model tiny_10x10 --test

# Compare multiple models
python watch_minimal_ai.py --compare tiny_10x10 medium_64x64 large_128x128
```

### Train Your Own AI

```bash
# Basic training with defaults
python train_minimal_ai.py

# Specify architecture and steps
python train_minimal_ai.py --arch 64 64 --steps 250000

# Use GPU
python train_minimal_ai.py --arch 64 64 --steps 500000 --device cuda

# Custom model name
python train_minimal_ai.py --arch 28 28 --steps 250000 --name my_model
```

### Run Multiple Experiments

```bash
# Run default experiment set (8 architectures)
python train_multiple_minimal_ai.py

# Run experiments in parallel
python train_multiple_minimal_ai.py --parallel 4

# Quick test with fewer steps
python train_multiple_minimal_ai.py --experiments quick

# See live output (sequential)
python train_multiple_minimal_ai.py --live

# Preview without running
python train_multiple_minimal_ai.py --dry-run
```

## Training Scripts

### Configurable Training (`train_minimal_ai.py`)

Train a single model with configurable parameters:

```bash
python train_minimal_ai.py [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--arch N N ...` | Network architecture (e.g., `64 64` or `128 128 128`) | `64 64` |
| `--steps N` | Number of training steps | `250000` |
| `--device` | `cpu` or `cuda` | `cpu` |
| `--name` | Model name for saving | Auto-generated |

**Examples:**
```bash
python train_minimal_ai.py --arch 10 10 --steps 100000 --name tiny
python train_minimal_ai.py --arch 64 64 64 --steps 500000 --device cuda --name deep
python train_minimal_ai.py --arch 128 128 --steps 250000 --name large
```

### Multi-Experiment Training (`train_multiple_minimal_ai.py`)

Run multiple experiments and automatically compare results:

```bash
python train_multiple_minimal_ai.py [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--experiments` | Experiment set: `default`, `quick`, `depth`, `width` | `default` |
| `--parallel N` | Run N experiments in parallel | `2` |
| `--steps N` | Steps per experiment | `250000` |
| `--device` | `cpu` or `cuda` | `cpu` |
| `--live` | Show live output (sequential) | Off |
| `--no-compare` | Skip final model comparison | Off |
| `--test-episodes N` | Episodes for comparison | `20` |
| `--dry-run` | Preview without running | Off |

**Experiment Sets:**

| Set | Experiments | Description |
|-----|-------------|-------------|
| `default` | 8 | Tiny, small, medium, large, and deep networks |
| `quick` | 3 | Fast test (50k steps): 10x10, 28x28, 64x64 |
| `depth` | 5 | Compare depths: 1-5 layers of 64 neurons |
| `width` | 5 | Compare widths: 16 to 256 neurons |

**Examples:**
```bash
# Run all default experiments with 4 parallel processes
python train_multiple_minimal_ai.py --parallel 4

# Quick test on GPU
python train_multiple_minimal_ai.py --experiments quick --device cuda

# Compare network depths with more training
python train_multiple_minimal_ai.py --experiments depth --steps 500000

# Watch live output
python train_multiple_minimal_ai.py --experiments quick --live
```

### Monitoring Parallel Experiments

When running experiments in parallel, monitor progress with:

```bash
# Watch all experiment logs in real-time
tail -f logs/experiments/*.log

# Watch a specific experiment
tail -f logs/experiments/tiny_10x10.log

# Use TensorBoard for training curves
tensorboard --logdir logs/tensorboard/
```

## Watching & Comparing Models

### Watch Script (`watch_minimal_ai.py`)

```bash
python watch_minimal_ai.py [options]
```

| Option | Description |
|--------|-------------|
| `--model NAME` | Load specific model |
| `--list` | List all available models |
| `--test` | Test performance (no graphics) |
| `--episodes N` | Number of test episodes |
| `--compare M1 M2 ...` | Compare multiple models |
| `--speed S` | Playback speed (seconds per frame) |
| `--show-obs` | Show observation features |

**Examples:**
```bash
# List available models
python watch_minimal_ai.py --list

# Watch specific model
python watch_minimal_ai.py --model medium_64x64

# Test model performance
python watch_minimal_ai.py --model medium_64x64 --test --episodes 50

# Compare models head-to-head
python watch_minimal_ai.py --compare tiny_10x10 medium_64x64 large_128x128

# Faster/slower playback
python watch_minimal_ai.py --model tiny_10x10 --speed 0.05
```

## Training Environments

### 1. Minimal Environment (Recommended)

**Best for fast training and experimentation.**

- **Observation Space:** 20 dimensions (focused features only)
- **Features:** Danger indicators, food direction, wall distances, snake length, body proximity
- **Training Speed:** ~10x faster than full board environments

### 2. Simple Environment

- **Observation Space:** 777 dimensions (full board + metadata)
- **Features:** Board representation, reward balance, direction, food quadrant, danger indicators

### 3. Advanced Environment

**For advanced users - includes curriculum learning.**

- **Observation Space:** 768 dimensions (flattened grid)
- **Features:** Curriculum learning (1→10 foods), hunger system, exploration rewards

## Neural Network Architectures

Common architecture configurations:

| Architecture | Layers | Parameters | Training Speed | Use Case |
|-------------|--------|------------|----------------|----------|
| `[10, 10]` | 2 | ~200 | Fastest | Quick tests |
| `[28, 28]` | 2 | ~1.5K | Fast | Lightweight |
| `[64, 64]` | 2 | ~8K | Medium | **Recommended** |
| `[64, 64, 64]` | 3 | ~12K | Medium | Deeper learning |
| `[128, 128]` | 2 | ~33K | Slower | More capacity |

## Monitoring Training

### TensorBoard

```bash
# View all experiments
tensorboard --logdir logs/tensorboard/

# View specific experiment
tensorboard --logdir logs/tensorboard/minimal_64x64/
```

Then open http://localhost:6006 in your browser.

### Experiment Logs

```bash
# Real-time log monitoring
tail -f logs/experiments/*.log
```

## GPU Training

The training scripts support NVIDIA GPU acceleration:

```bash
# Single experiment
python train_minimal_ai.py --arch 64 64 --device cuda

# Multiple experiments
python train_multiple_minimal_ai.py --device cuda --parallel 2
```

**Requirements:**
- NVIDIA GPU with CUDA support
- PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Game Controls

### Human Play (`snake_game.py`)
- **Arrow Keys:** Move the snake
- **R:** Restart after game over

### AI Visualization
- Close the window to stop
- `Ctrl+C` to stop from terminal

## Technical Details

### Algorithm
- **PPO** (Proximal Policy Optimization) from Stable Baselines3

### Game Grid
- **Window Size:** 640×480 pixels
- **Cell Size:** 20×20 pixels
- **Grid Size:** 32×24 cells

### Action Space

| Action | Direction |
|--------|-----------|
| 0 | Up |
| 1 | Right |
| 2 | Down |
| 3 | Left |

### Reward System (Minimal Environment)

| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Move toward food | +0.5 per step closer |
| Move away from food | -0.5 per step farther |
| Each step | -0.01 |
| Death | -1 × snake_length (dynamic) |

## Troubleshooting

### "No trained model found!"

Train a model first:
```bash
python train_minimal_ai.py --arch 64 64 --steps 100000
```

### ImportError or ModuleNotFoundError

Ensure you're using Python 3.11 and have installed all dependencies:
```bash
python3.11 -m pip install -r requirements.txt
```

### Pygame window not showing

On headless systems:
```bash
export SDL_VIDEODRIVER=dummy
```

### Training is slow

1. Use the minimal environment (20 features vs 777)
2. Enable GPU: `--device cuda`
3. Use smaller architectures: `--arch 28 28`

## Dependencies

- `gymnasium==1.0.0` - RL environment interface
- `stable_baselines3==2.4.1` - PPO implementation
- `pygame==2.6.1` - Game rendering
- `numpy==1.23.5` - Numerical operations
- `tensorboard==2.18.0` - Training visualization
- `matplotlib==3.10.5` - Plotting
- `tqdm==4.66.6` - Progress bars
- `rich==13.9.4` - Console formatting

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues and pull requests for improvements!
