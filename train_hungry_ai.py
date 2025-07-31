#!/usr/bin/env python3
"""
Train AI with HUNGER/STARVATION system
No more safe circling - AI must hunt for food!
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

def main():
    print("🍽️  TRAINING HUNGRY AI (NO MORE CIRCLING!)")
    print("=" * 50)
    
    # Create hungry environment
    class HungrySnakeEnv(SnakeEnvironment):
        def __init__(self):
            super().__init__()
            self.max_steps = 200  # Shorter episodes (hunger pressure)
            self.step_count = 0
            
        def reset(self, seed=None):
            self.step_count = 0
            return super().reset(seed)
            
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.step_count += 1
            
            # Force episode end if too long (prevent infinite wandering)
            if self.step_count >= self.max_steps:
                truncated = True
                # Extra penalty for taking too long
                reward -= 2
                
            return obs, reward, terminated, truncated, info
    
    env = HungrySnakeEnv()
    env = Monitor(env, "logs_hungry/")
    
    print(f"✅ Hungry environment ready!")
    print(f"   Max episode length: 200 steps")
    print(f"   Hunger system: ACTIVE")
    print(f"   Circling strategy: IMPOSSIBLE")
    
    # Create or load model
    print(f"\n🧠 Setting up neural network...")
    
    try:
        # Try to load existing model
        model = PPO.load("models/snake_ai_simple", env=env, device="cpu")
        print(f"✅ Loaded existing model (will retrain with hunger!)")
    except:
        # Create new model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0005,  # Slightly higher learning rate
            n_steps=1024,
            batch_size=64,
            device="cpu",
        )
        print(f"✅ Created new hungry AI!")
    
    # Train the hungry AI
    print(f"\n🏋️ Training hungry AI...")
    print(f"AI must learn to find food quickly or starve!")
    
    # Test current performance first
    print(f"\n📊 BEFORE HUNGER TRAINING:")
    test_ai_performance(model, env, "BEFORE")
    
    # Train with hunger pressure
    model.learn(
        total_timesteps=30000,  # More training for harder problem
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    print(f"✅ Hungry AI training completed!")
    
    # Save the hungry model
    print(f"\n💾 Saving hungry AI model...")
    model.save("models/hungry_snake_ai")
    print(f"✅ Saved as 'hungry_snake_ai'")
    
    # Test final performance
    print(f"\n📊 AFTER HUNGER TRAINING:")
    test_ai_performance(model, env, "AFTER")
    
    print(f"\n🎉 HUNGRY AI TRAINING COMPLETE!")
    print(f"Your AI should now be much better at finding food!")

def test_ai_performance(model, env, phase):
    """Test AI performance with hunger system"""
    total_scores = []
    total_steps = []
    total_max_hunger = []
    
    print(f"Testing {phase} hunger training (5 episodes)...")
    
    for episode in range(5):
        obs, info = env.reset()
        episode_score = 0
        episode_steps = 0
        max_hunger = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            max_hunger = max(max_hunger, info['hunger'])
            
            if terminated or truncated:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        total_max_hunger.append(max_hunger)
        
        print(f"   Episode {episode+1}: Score={episode_score}, Steps={episode_steps}, Max Hunger={max_hunger}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    avg_hunger = sum(total_max_hunger) / len(total_max_hunger)
    
    print(f"📈 {phase} RESULTS:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Max Hunger: {avg_hunger:.1f}")
    
    if avg_score > 1:
        print(f"   🎉 EXCELLENT: Finding multiple foods!")
    elif avg_score > 0:
        print(f"   🎯 GOOD: Finding food consistently!")
    else:
        print(f"   😰 STRUGGLING: Still learning to hunt")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_hungry", exist_ok=True)
    
    main()