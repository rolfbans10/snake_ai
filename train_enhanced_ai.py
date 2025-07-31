#!/usr/bin/env python3
"""
Train AI with Enhanced Reward System
- Distance rewards for directional guidance  
- Configurable reward parameters
- Multiple pressure systems
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

def main():
    print("🚀 TRAINING WITH ENHANCED REWARD SYSTEM!")
    print("=" * 50)
    
    # Create environment and show configuration
    base_env = SnakeEnvironment()
    env = Monitor(base_env, "logs_enhanced/")
    
    print(f"📊 REWARD CONFIGURATION:")
    print(f"   💀 Death Penalty: {base_env.DEATH_PENALTY}")
    print(f"   🍎 Food Reward: {base_env.FOOD_REWARD}")
    print(f"   ⏱️ Step Penalty: {base_env.STEP_PENALTY}")
    print(f"   📏 Distance: +{base_env.DISTANCE_REWARD} closer / {base_env.DISTANCE_PENALTY} farther")
    print(f"   😰 Hunger Max: {base_env.HUNGER_MAX} steps")
    print("=" * 50)
    
    # Create fresh AI model
    print(f"🧠 Creating fresh AI model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        device="cpu",
    )
    
    print(f"✅ Model created!")
    
    # Train with enhanced rewards
    print(f"\n🏋️ TRAINING SESSION:")
    print(f"Training for 2,500,000 steps with enhanced rewards...")
    
    model.learn(
        total_timesteps=2500000,
        progress_bar=True
    )
    
    print(f"✅ Training completed!")
    
    # Save the model
    print(f"\n💾 Saving enhanced AI...")
    model.save("models/enhanced_reward_ai")
    print(f"✅ Saved as 'enhanced_reward_ai'")
    
    # Quick test
    print(f"\n🧪 QUICK PERFORMANCE TEST:")
    test_episodes = 3
    total_scores = []
    total_steps = []
    
    for episode in range(test_episodes):
        obs, info = env.reset()
        episode_steps = 0
        episode_score = 0
        
        while episode_steps < 150:  # Limit episode length
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        
        print(f"   Episode {episode+1}: Score={episode_score}, Steps={episode_steps}")
    
    avg_score = np.mean(total_scores)
    avg_steps = np.mean(total_steps)
    
    print(f"\n📈 RESULTS:")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Total Foods Found: {sum(total_scores)}")
    
    if avg_score > 0.5:
        print(f"   🏆 EXCELLENT: AI is finding food consistently!")
    elif avg_score > 0:
        print(f"   🎯 GOOD: AI found some food!")
    elif avg_steps > 50:
        print(f"   📈 PROMISING: AI is exploring (not dying quickly)!")
    else:
        print(f"   🔧 LEARNING: AI needs more training")
    
    print(f"\n🎉 ENHANCED TRAINING COMPLETE!")
    print(f"The AI now has:")
    print(f"   - Distance guidance ✅")
    print(f"   - Hunger pressure ✅") 
    print(f"   - Death avoidance ✅")
    print(f"   - Efficiency pressure ✅")
    print(f"   - Big food rewards ✅")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_enhanced", exist_ok=True)
    
    main()