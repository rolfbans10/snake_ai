#!/usr/bin/env python3
"""
Simple Neural Network Training Script for Snake AI
Training script for the simplified Snake environment with basic reward system
"""

from snake_env_simple import SimpleSnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

def main(training_steps=250000):
    print("🧠 SIMPLE NEURAL NETWORK TRAINING FOR SNAKE AI")
    print("=" * 55)
    print(f"🎯 Training for {training_steps:,} steps")
    
    # STEP 1: Create the training environment
    print("\n📋 STEP 1: Setting up simple environment...")
    env = SimpleSnakeEnvironment()
    
    # Wrap with Monitor to track statistics
    os.makedirs("logs/simple", exist_ok=True)
    env = Monitor(env, "logs/simple/")
    
    print(f"✅ Simple environment ready!")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Reward system: +10 food, -1 death, -0.1 step, +10 start balance")
    
    # STEP 2: Create the AI agent (Neural Network)
    print("\n🧠 STEP 2: Creating neural network...")
    
    # PPO = Proximal Policy Optimization
    # Adjusted parameters for simple reward system
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron
        env,
        verbose=1,    # Print training progress
        learning_rate=0.001,   # Slightly higher learning rate for simple rewards
        n_steps=2048,          # Steps per training batch
        batch_size=64,         # Training batch size
        n_epochs=10,           # Training epochs per batch
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE lambda
        clip_range=0.2,        # PPO clip range
        device="cuda",          # Use CPU (change to "cuda" if you have GPU)
        tensorboard_log="./logs/tensorboard/simple/"
    )
    
    print(f"✅ Neural network created!")
    print(f"   Policy: {model.policy}")
    print(f"   Device: {model.device}")
    
    # STEP 3: Test random agent first (baseline)
    print("\n🎲 STEP 3: Testing random agent (baseline)...")
    test_random_agent(env, episodes=5)
    
    # STEP 4: Train the neural network
    print("\n🏋️ STEP 4: Training neural network...")
    print("This will take several minutes...")
    
    # Create evaluation environment for monitoring progress
    os.makedirs("logs/simple/eval", exist_ok=True)
    eval_env = Monitor(SimpleSnakeEnvironment(), "logs/simple/eval/")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="./models/",
        log_path="./logs/simple/", 
        eval_freq=10000,      # Evaluate every 10k steps
        deterministic=True, 
        render=False,
        n_eval_episodes=10    # Run 10 episodes for evaluation
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
    model.save("models/simple_snake_ai")
    print("✅ Model saved as 'models/simple_snake_ai'")
    
    # STEP 6: Test the trained AI
    print("\n🧪 STEP 6: Testing trained AI...")
    test_trained_agent(model, env, episodes=10)
    
    print("\n🎉 SIMPLE TRAINING COMPLETE!")
    print("Your simple AI snake is ready! 🐍🧠")
    print("Run 'python watch_simple_ai.py' to watch it play!")

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
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            episode_reward += reward
            
            if terminated or truncated:
                episode_score = info['score']
                break
                
            # Prevent infinite episodes
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
            # AI chooses action (no randomness)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                episode_score = info['score']
                break
                
            # Prevent infinite episodes
            if episode_steps > 2000:  # Higher limit for trained agent
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
    
    # Performance comparison
    if avg_score > 3:
        print("🏆 EXCELLENT! Your AI learned to play Snake well!")
    elif avg_score > 1:
        print("👍 GOOD! Your AI is learning to play Snake!")
    elif avg_score > 0:
        print("📈 PROGRESS! Your AI is starting to learn!")
    else:
        print("🔄 Keep training - the AI needs more time to learn!")

if __name__ == "__main__":
    import sys
    
    # Show help message first
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("🐍 Simple Snake AI Training Script")
        print("Usage: python train_simple_ai.py [steps]")
        print("")
        print("Arguments:")
        print("  steps    Number of training steps (default: 250,000)")
        print("")
        print("Examples:")
        print("  python train_simple_ai.py          # Default training")
        print("  python train_simple_ai.py 100000   # Quick training")
        print("  python train_simple_ai.py 500000   # Extended training")
        print("  python train_simple_ai.py 1000000  # Long training")
        sys.exit(0)
    
    # Parse command line arguments
    training_steps = 250000  # Default value
    
    if len(sys.argv) > 1:
        try:
            training_steps = int(sys.argv[1])
            if training_steps <= 0:
                raise ValueError("Training steps must be positive")
        except ValueError as e:
            print(f"❌ Error: Invalid training steps argument - {e}")
            print("Usage: python train_simple_ai.py [steps]")
            print("Examples:")
            print("  python train_simple_ai.py          # Train for 250,000 steps (default)")
            print("  python train_simple_ai.py 100000   # Train for 100,000 steps")
            print("  python train_simple_ai.py 500000   # Train for 500,000 steps")
            sys.exit(1)
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/simple", exist_ok=True)
    os.makedirs("logs/simple/eval", exist_ok=True)
    os.makedirs("logs/tensorboard/simple", exist_ok=True)
    
    main(training_steps)