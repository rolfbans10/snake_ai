#!/usr/bin/env python3
"""
Minimal Neural Network Training Script for Snake AI
Training script for the minimal Snake environment with focused feature set (no board)

Usage:
    python train_minimal_ai.py --arch 64 64 --steps 250000 --device cpu
    python train_minimal_ai.py --arch 28 28 28 --steps 500000 --device cuda
    python train_minimal_ai.py --arch 10 10 --steps 100000 --name tiny_model
"""

from snake_env_simple_minimal import MinimalSnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import argparse
import os


def main(training_steps: int, arch: list, device: str, model_name: str):
    # Create architecture string for display
    arch_str = "x".join(map(str, arch))
    
    print("🧠 MINIMAL NEURAL NETWORK TRAINING FOR SNAKE AI")
    print("=" * 60)
    print(f"🎯 Training for {training_steps:,} steps")
    print(f"🏗️  Architecture: {arch} ({arch_str})")
    print(f"💻 Device: {device}")
    print(f"💾 Model name: {model_name}")
    print("🚀 Using MINIMAL feature set (20 dims) - much faster training!")
    
    # Create log directory based on model name (organized under logs/)
    log_dir = f"logs/minimal_{model_name}"
    tensorboard_dir = f"logs/tensorboard/minimal_{model_name}"
    
    # STEP 1: Create the training environment
    print("\n📋 STEP 1: Setting up minimal environment...")
    env = MinimalSnakeEnvironment()
    
    # Wrap with Monitor to track statistics
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, f"{log_dir}/")
    
    print(f"✅ Minimal environment ready!")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Reward system: +10 food, -1 death, -0.01 step, ±0.5 direction")
    
    # STEP 2: Create the AI agent (Neural Network)
    print("\n🧠 STEP 2: Creating neural network...")
    
    # PPO = Proximal Policy Optimization
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device=device,
        tensorboard_log=f"./{tensorboard_dir}/",
        policy_kwargs=dict(
            net_arch=arch
        )
    )
    
    print(f"✅ Neural network created!")
    print(f"   Policy: {model.policy}")
    print(f"   Device: {model.device}")
    print(f"   Network: {arch}")
    
    # STEP 3: Test random agent first (baseline)
    print("\n🎲 STEP 3: Testing random agent (baseline)...")
    test_random_agent(env, episodes=5)
    
    # STEP 4: Train the neural network
    print("\n🏋️ STEP 4: Training neural network...")
    print("This should be MUCH faster than the full board version...")
    
    # Create evaluation environment for monitoring progress
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    eval_env = Monitor(MinimalSnakeEnvironment(), f"{log_dir}/eval/")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="./models/",
        log_path=f"./{log_dir}/", 
        eval_freq=5000,
        deterministic=True, 
        render=False,
        n_eval_episodes=10
    )
    
    # Train with specified number of steps
    model.learn(
        total_timesteps=training_steps,
        callback=eval_callback,
        progress_bar=True
    )
    
    print("✅ Training completed!")
    
    # STEP 5: Save the trained model
    print("\n💾 STEP 5: Saving trained model...")
    model_path = f"models/{model_name}"
    model.save(model_path)
    print(f"✅ Model saved as '{model_path}'")
    
    # STEP 6: Test the trained AI
    print("\n🧪 STEP 6: Testing trained AI...")
    test_trained_agent(model, env, episodes=10)
    
    print("\n🎉 MINIMAL TRAINING COMPLETE!")
    print(f"📁 Model: models/{model_name}.zip")
    print(f"📊 Logs: {log_dir}/")
    print(f"📈 TensorBoard: tensorboard --logdir logs/tensorboard/")
    print(f"\nWatch it play: python watch_minimal_ai.py --model {model_name}")


def test_random_agent(env, episodes=5):
    """Test how well a random agent performs"""
    print(f"Running {episodes} episodes with random actions...")
    
    total_scores = []
    total_steps = []
    total_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_score = 0
        episode_steps = 0
        episode_reward = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            episode_reward += reward
            
            if terminated or truncated:
                episode_score = info['score']
                break
                
            if episode_steps > 1000:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        total_rewards.append(info['reward_balance'])
        print(f"   Episode {episode+1}: Score = {episode_score}, Steps = {episode_steps}, Final Balance = {info['reward_balance']:.1f}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    avg_reward = sum(total_rewards) / len(total_rewards)
    
    print(f"📊 Random Agent Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Final Balance: {avg_reward:.1f}")
    print(f"   Best Score: {max(total_scores)}")


def test_trained_agent(model, env, episodes=10):
    """Test how well the trained agent performs"""
    print(f"Running {episodes} episodes with trained AI...")
    
    total_scores = []
    total_steps = []
    total_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_score = 0
        episode_steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                episode_score = info['score']
                break
                
            if episode_steps > 2000:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        total_rewards.append(info['reward_balance'])
        print(f"   Episode {episode+1}: Score = {episode_score}, Steps = {episode_steps}, Final Balance = {info['reward_balance']:.1f}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    avg_reward = sum(total_rewards) / len(total_rewards)
    
    print(f"📊 Trained AI Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Final Balance: {avg_reward:.1f}")
    print(f"   Best Score: {max(total_scores)}")
    
    if avg_score > 5:
        print("🏆 EXCELLENT! Your minimal AI learned to play Snake amazingly!")
    elif avg_score > 3:
        print("👍 GREAT! Your minimal AI is playing Snake well!")
    elif avg_score > 1:
        print("📈 GOOD! Your minimal AI is learning to play Snake!")
    else:
        print("🔄 Keep training - the AI needs more time to learn!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🐍 Minimal Snake AI Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_minimal_ai.py --arch 64 64 --steps 250000 --device cpu
  python train_minimal_ai.py --arch 28 28 --steps 250000 --device cuda
  python train_minimal_ai.py --arch 10 10 --steps 100000 --name tiny
  python train_minimal_ai.py --arch 64 64 64 --steps 500000 --name deep

Run multiple experiments in parallel:
  python train_minimal_ai.py --arch 10 10 --steps 250000 --name arch_10x10 &
  python train_minimal_ai.py --arch 28 28 --steps 250000 --name arch_28x28 &
  python train_minimal_ai.py --arch 64 64 --steps 250000 --name arch_64x64 &
        """
    )
    
    parser.add_argument(
        "--arch", 
        type=int, 
        nargs="+", 
        default=[64, 64],
        help="Network architecture as space-separated integers (default: 64 64)"
    )
    
    parser.add_argument(
        "--steps", 
        type=int, 
        default=250000,
        help="Number of training steps (default: 250000)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to train on: cpu or cuda (default: cpu)"
    )
    
    parser.add_argument(
        "--name", 
        type=str, 
        default=None,
        help="Model name for saving (default: minimal_<arch>)"
    )
    
    args = parser.parse_args()
    
    # Generate model name from architecture if not provided
    if args.name is None:
        arch_str = "x".join(map(str, args.arch))
        args.name = f"minimal_{arch_str}"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    
    # Run training
    main(
        training_steps=args.steps,
        arch=args.arch,
        device=args.device,
        model_name=args.name
    )
