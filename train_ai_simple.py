#!/usr/bin/env python3
"""
SIMPLE Fixed Training - No hanging issues!
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

def main():
    print("🧠 SIMPLE FIXED TRAINING (NO HANGS!)")
    print("=" * 50)
    
    # STEP 1: Create environment with episode length limit
    print("\n📋 STEP 1: Setting up environment...")
    
    class LimitedSnakeEnv(SnakeEnvironment):
        def __init__(self):
            super().__init__()
            self.max_steps = 300  # Limit episodes to 300 steps
            self.step_count = 0
            
        def reset(self, seed=None):
            self.step_count = 0
            return super().reset(seed)
            
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.step_count += 1
            
            # Force episode to end if too long (prevents infinite wandering)
            if self.step_count >= self.max_steps:
                truncated = True
                # Small penalty for taking too long
                reward -= 1
                
            return obs, reward, terminated, truncated, info
    
    env = LimitedSnakeEnv()
    env = Monitor(env, "logs_simple/")
    
    print(f"✅ Environment ready with 300-step limit!")
    
    # STEP 2: Create or load model
    print("\n🧠 STEP 2: Loading existing model...")
    
    try:
        model = PPO.load("models/best_model", env=env, device="cpu")
        print(f"✅ Loaded existing model to continue training!")
        print(f"   Will continue from where training left off")
    except:
        print(f"❌ Could not load model, creating new one...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            device="cpu",
        )
    
    # STEP 3: Quick training session
    print("\n🏋️ STEP 3: Quick training session...")
    print("Training for 25,000 steps (should take 2-3 minutes)")
    
    model.learn(
        total_timesteps=25000,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    print("✅ Training completed!")
    
    # STEP 4: Save model
    print("\n💾 STEP 4: Saving model...")
    model.save("models/snake_ai_simple")
    print("✅ Model saved!")
    
    # STEP 5: Test the model
    print("\n🧪 STEP 5: Testing the trained AI...")
    test_ai(model, env)
    
    print("\n🎉 SIMPLE TRAINING COMPLETE!")

def test_ai(model, env):
    """Test the AI performance"""
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
    
    print(f"\n📊 AI PERFORMANCE:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Best Score: {max(total_scores)}")
    
    print(f"\n🏆 COMPARISON:")
    print(f"   Random Agent: ~25 steps, 0 score")
    print(f"   Before Training: ~34 steps, 0 score")
    print(f"   After Training: ~{avg_steps:.0f} steps, {avg_score:.1f} score")
    
    if avg_score > 0:
        print(f"   🎉 AMAZING: AI learned to find food!")
    elif avg_steps > 50:
        print(f"   🎯 GOOD: AI learned to survive much longer!")
    else:
        print(f"   🤔 Still learning: Need more training time")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_simple", exist_ok=True)
    
    main()