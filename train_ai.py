#!/usr/bin/env python3
"""
Neural Network Training Script for Snake AI
Step-by-step guide to training an AI agent to play Snake!
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

def main():
    print("🧠 NEURAL NETWORK TRAINING FOR SNAKE AI")
    print("=" * 50)
    
    # STEP 1: Create the training environment
    print("\n📋 STEP 1: Setting up environment...")
    env = SnakeEnvironment()
    
    # Wrap with Monitor to track statistics
    env = Monitor(env, "logs/")
    
    print(f"✅ Environment ready!")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # STEP 2: Create the AI agent (Neural Network)
    print("\n🧠 STEP 2: Creating neural network...")
    
    # PPO = Proximal Policy Optimization (good for beginners)
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron (simple neural network)
        env,
        verbose=1,    # Print training progress
        learning_rate=0.0003,  # How fast the AI learns
        n_steps=2048,          # Steps per training batch
        batch_size=64,         # Training batch size
        device="cpu",          # Use CPU (change to "cuda" if you have GPU)
        tensorboard_log="./tensorboard_logs/"
    )
    
    print(f"✅ Neural network created!")
    print(f"   Policy: {model.policy}")
    print(f"   Device: {model.device}")
    
    # STEP 3: Test random agent first (baseline)
    print("\n🎲 STEP 3: Testing random agent (baseline)...")
    test_random_agent(env, episodes=5)
    
    # STEP 4: Train the neural network
    print("\n🏋️ STEP 4: Training neural network...")
    print("This will take a few minutes...")
    
    # Create evaluation environment for monitoring progress
    eval_env = Monitor(SnakeEnvironment(), "logs/eval/")
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="./models/",
        log_path="./logs/", 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )
    
    # Train for 100,000 steps
    model.learn(
        total_timesteps=100000,
        callback=eval_callback,
        progress_bar=True
    )
    
    print("✅ Training completed!")
    
    # STEP 5: Save the trained model
    print("\n💾 STEP 5: Saving trained model...")
    model.save("models/snake_ai_model")
    print("✅ Model saved as 'models/snake_ai_model'")
    
    # STEP 6: Test the trained AI
    print("\n🧪 STEP 6: Testing trained AI...")
    test_trained_agent(model, env, episodes=5)
    
    print("\n🎉 TRAINING COMPLETE!")
    print("Your AI snake is ready! 🐍🧠")

def test_random_agent(env, episodes=5):
    """Test how well a random agent performs"""
    print(f"Running {episodes} episodes with random actions...")
    
    total_scores = []
    total_steps = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_score = 0
        episode_steps = 0
        
        while True:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                episode_score = info['score']
                break
                
            # Prevent infinite episodes
            if episode_steps > 1000:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        print(f"   Episode {episode+1}: Score = {episode_score}, Steps = {episode_steps}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    
    print(f"📊 Random Agent Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Best Score: {max(total_scores)}")

def test_trained_agent(model, env, episodes=5):
    """Test how well the trained agent performs"""
    print(f"Running {episodes} episodes with trained AI...")
    
    total_scores = []
    total_steps = []
    
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
            if episode_steps > 1000:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        print(f"   Episode {episode+1}: Score = {episode_score}, Steps = {episode_steps}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    
    print(f"📊 Trained AI Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Best Score: {max(total_scores)}")

if __name__ == "__main__":
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    main()