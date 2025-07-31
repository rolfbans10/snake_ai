#!/usr/bin/env python3
"""
IMPROVED Neural Network Training - Fixed Version
Solves the "stuck at evaluation" problem
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

# Create a wrapper to limit episode length
class TimeLimitWrapper:
    def __init__(self, env, max_episode_steps=500):
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Force episode to end if too long
        if self.step_count >= self.max_episode_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)

def create_env():
    """Create a single training environment with time limits"""
    env = SnakeEnvironment()
    env = TimeLimitWrapper(env, max_episode_steps=500)  # Max 500 steps per episode
    return env

def main():
    print("🧠 IMPROVED NEURAL NETWORK TRAINING")
    print("=" * 50)
    
    # STEP 1: Create environment
    print("\n📋 STEP 1: Setting up IMPROVED environment...")
    
    # Create vectorized environment (faster training)
    env = make_vec_env(create_env, n_envs=4)  # Train on 4 environments in parallel
    
    print(f"✅ Environment ready with time limits!")
    
    # STEP 2: Create or load model
    print("\n🧠 STEP 2: Creating/Loading neural network...")
    
    model_path = "models/best_model"
    
    try:
        # Try to load existing model to continue training
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"✅ Loaded existing model to continue training!")
    except:
        # Create new model if loading fails
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,  # Smaller batches for faster updates
            batch_size=64,
            device="cpu",
            tensorboard_log="./tensorboard_logs/"
        )
        print(f"✅ Created new neural network!")
    
    # STEP 3: Train with NO evaluation callback (much faster)
    print("\n🏋️ STEP 3: Training neural network (NO EVALUATION HANGS)...")
    print("This should be much faster now!")
    
    # Train for shorter bursts
    total_steps = 50000  # Reduced from 100k for faster results
    
    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        reset_num_timesteps=False  # Continue from where we left off
    )
    
    print("✅ Training completed!")
    
    # STEP 4: Save model
    print("\n💾 STEP 4: Saving improved model...")
    model.save("models/snake_ai_improved")
    print("✅ Model saved!")
    
    # STEP 5: Test the improved AI
    print("\n🧪 STEP 5: Testing improved AI...")
    test_improved_ai(model)
    
    print("\n🎉 IMPROVED TRAINING COMPLETE!")

def test_improved_ai(model):
    """Test the improved model"""
    env = create_env()
    
    total_scores = []
    total_steps = []
    
    print("Running 5 test episodes...")
    
    for episode in range(5):
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
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        print(f"   Episode {episode+1}: Score = {episode_score}, Steps = {episode_steps}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    
    print(f"\n📊 IMPROVED AI Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Best Score: {max(total_scores)}")
    
    if avg_score > 0:
        print(f"   🎉 SUCCESS: AI found food!")
    elif avg_steps > 50:
        print(f"   🎯 PROGRESS: AI survives longer than random!")
    else:
        print(f"   🤔 Still learning: Need more training")

if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    main()