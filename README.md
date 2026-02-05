# Snake AI - Reinforcement Learning

A deep reinforcement learning project that trains an AI agent to play the classic Snake game using Proximal Policy Optimization (PPO) from Stable Baselines3.

## Requirements

> **IMPORTANT:** This project requires **Python 3.11**. Other Python versions are not supported and may cause compatibility issues with the dependencies.

## Features

- Classic Snake game implementation with Pygame
- Multiple training environments with different complexity levels
- Curriculum learning with progressive difficulty
- Real-time visualization of trained AI playing
- TensorBoard integration for training monitoring
- Pre-trained model included

## Project Structure

```
snake_ai/
├── snake_game.py              # Base Snake game with Pygame rendering
├── snake_env.py               # Advanced environment (curriculum learning)
├── snake_env_simple.py        # Simple environment (777-dim observations)
├── snake_env_simple_minimal.py # Minimal environment (20-dim observations)
├── train_ai.py                # Training script for advanced environment
├── train_simple_ai.py         # Training script for simple environment
├── train_minimal_ai.py        # Training script for minimal environment
├── watch_ai_play.py           # Visualize trained AI (advanced)
├── watch_simple_ai.py         # Visualize trained AI (simple)
├── watch_minimal_ai.py        # Visualize trained AI (minimal)
├── models/                    # Saved trained models
│   └── simple_snake_ai.zip    # Pre-trained model
├── requirements.txt           # Python dependencies
└── test_*.py                  # Various test scripts
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
# Watch with visualization
python watch_simple_ai.py

# Watch minimal AI with visualization
python watch_minimal_ai.py

# Test performance without visualization (20 episodes)
python watch_minimal_ai.py --test

# Show observation feature breakdown
python watch_minimal_ai.py --show-obs
```

### Train Your Own AI

```bash
# Quick training (recommended for minimal environment)
python train_minimal_ai.py 100000

# Standard training
python train_simple_ai.py 250000

# Extended training for better results
python train_simple_ai.py 500000
```

## Training Environments

### 1. Minimal Environment (`snake_env_simple_minimal.py`)

**Recommended for fast training and best results.**

- **Observation Space:** 20 dimensions (focused features only)
- **Features:** Danger indicators, food direction, wall distances, snake length, body proximity
- **Training Speed:** ~10x faster than full board environments

```bash
python train_minimal_ai.py [steps]
```

### 2. Simple Environment (`snake_env_simple.py`)

- **Observation Space:** 777 dimensions (full board + metadata)
- **Features:** Board representation, reward balance, direction, food quadrant, danger indicators

```bash
python train_simple_ai.py [steps]
```

### 3. Advanced Environment (`snake_env.py`)

**For advanced users - includes curriculum learning.**

- **Observation Space:** 768 dimensions (flattened grid)
- **Features:** Curriculum learning (1→10 foods), hunger system, exploration rewards, win condition
- **Best For:** Complex training scenarios

```bash
python train_ai.py
```

## Reward Systems

### Minimal/Simple Environment Rewards

| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Move toward food | +0.5 per step closer |
| Move away from food | -0.5 per step farther |
| Each step | -0.01 |
| Death | -1 × snake_length (dynamic) |

### Advanced Environment Rewards

| Event | Reward |
|-------|--------|
| Eat food | +100 |
| Move toward food | +2.0 |
| Move away from food | -2.0 |
| Explore new area | +1.0 |
| Each step | -0.05 |
| Death | -200 |
| Win (eat all foods) | +1000 |
| Starvation (2500 steps without food) | Death |

## Monitoring Training

### TensorBoard

Training logs are saved to `tensorboard_logs_*` directories. View them with:

```bash
tensorboard --logdir tensorboard_logs_minimal/
# or
tensorboard --logdir tensorboard_logs_simple/
```

Then open http://localhost:6006 in your browser.

### Training Output

The training script displays:
- Episode scores and steps
- Food eaten events
- Death causes (wall collision, self collision, starvation)
- Model evaluation results

## Model Files

Trained models are saved in the `models/` directory:

- `simple_snake_ai.zip` - Simple environment model
- `minimal_snake_ai.zip` - Minimal environment model
- `best_model.zip` - Best performing model from evaluation callbacks

## Game Controls

### Human Play (`snake_game.py`)
- **Arrow Keys:** Move the snake
- **R:** Restart after game over

### AI Visualization

**Watch Scripts Parameters:**

| Script | Option | Description |
|--------|--------|-------------|
| `watch_simple_ai.py` | (none) | Watch AI play with visualization |
| `watch_simple_ai.py` | `--test` | Test performance (20 episodes, no graphics) |
| `watch_minimal_ai.py` | (none) | Watch AI play with visualization |
| `watch_minimal_ai.py` | `--test` | Test performance (20 episodes, no graphics) |
| `watch_minimal_ai.py` | `--show-obs` | Show observation feature breakdown |
| `watch_minimal_ai.py` | `--help` | Show help message |

**Controls:**
- Close the window to stop
- `Ctrl+C` to stop from terminal

## Technical Details

### Neural Network Architecture

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** MLP (Multi-Layer Perceptron)
- **Network:** 64×64 hidden layers (minimal), larger for full board
- **Learning Rate:** 0.003 (minimal), 0.001 (simple), 0.0003 (advanced)
- **Device:** CPU by default, GPU (CUDA) supported

### GPU Training

The training scripts default to CPU but support NVIDIA GPU acceleration:

1. Ensure you have an NVIDIA GPU with CUDA support
2. Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Edit the training script and change `device="cpu"` to `device="cuda"`

GPU training provides significant speedup for longer training runs.

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

## Troubleshooting

### "No trained model found!"

Train a model first:
```bash
python train_minimal_ai.py 100000
```

### ImportError or ModuleNotFoundError

Ensure you're using Python 3.11 and have installed all dependencies:
```bash
python3.11 -m pip install -r requirements.txt
```

### Pygame window not showing

Ensure you have a display available. On headless systems, you may need to set:
```bash
export SDL_VIDEODRIVER=dummy  # For training without display
```

### Training is slow

- Use the minimal environment (`train_minimal_ai.py`) - 10x faster than full board
- **Enable GPU acceleration:** Edit the training script and change `device="cpu"` to `device="cuda"`
  
  ```python
  # In train_minimal_ai.py, train_simple_ai.py, or train_ai.py
  model = PPO(
      ...
      device="cuda",  # Change from "cpu" to "cuda" for GPU training
      ...
  )
  ```
  
- Requires NVIDIA GPU with CUDA support and PyTorch with CUDA installed

## Testing AI Performance

Run performance tests without visualization:

```bash
# Test simple AI
python watch_simple_ai.py --test

# Test minimal AI
python watch_minimal_ai.py --test

# Show minimal AI observation details
python watch_minimal_ai.py --show-obs
```

This runs 20 episodes and reports:
- Average score and best score
- Average steps survived
- Average reward balance
- Performance rating comparison

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
