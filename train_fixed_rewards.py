#!/usr/bin/env python3
"""
Training with FIXED reward system - death -35, food +15
This should prevent the AI from learning to die quickly!
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class SmartProgressCallback(BaseCallback):
    """Track progress and catch reward hacking early"""
    
    def __init__(self, eval_freq=3000):
        super().__init__()
        self.eval_freq = eval_freq
        self.progress_data = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n🔍 SMART CHECK at {self.n_calls} steps:")
            scores, steps, food_rate, death_strategy = self.evaluate_with_analysis()
            
            self.progress_data.append({
                'steps': self.n_calls,
                'avg_score': scores,
                'avg_episode_length': steps,
                'food_success_rate': food_rate,
                'quick_death_rate': death_strategy
            })
            
            print(f"   📊 Average Score: {scores:.2f}")
            print(f"   📏 Average Episode Length: {steps:.1f}")
            print(f"   🍽️ Food Success Rate: {food_rate:.1%}")
            print(f"   ⚡ Quick Death Rate: {death_strategy:.1%}")
            
            # Detect reward hacking
            if death_strategy > 0.5:  # More than 50% quick deaths
                print(f"   ⚠️  WARNING: AI might be learning to die quickly!")
            elif food_rate > 0.3:  # Good food finding
                print(f"   ✅ GOOD: AI is actively seeking food!")
            elif steps > 50:  # Longer episodes
                print(f"   📈 PROMISING: AI is exploring more!")
                
        return True
    
    def evaluate_with_analysis(self):
        """Analyze AI behavior to detect reward hacking"""
        env = SnakeEnvironment()
        
        scores = []
        episode_lengths = []
        foods_found = 0
        quick_deaths = 0  # Episodes ending in < 30 steps
        
        for _ in range(5):  # Quick evaluation
            obs, info = env.reset()
            episode_steps = 0
            episode_score = 0
            
            while episode_steps < 200:  # Episode limit
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                if terminated or truncated:
                    episode_score = info['score']
                    break
            
            scores.append(episode_score)
            episode_lengths.append(episode_steps)
            
            if episode_score > 0:
                foods_found += 1
            if episode_steps < 30:  # Quick death detection
                quick_deaths += 1
        
        avg_score = np.mean(scores)
        avg_length = np.mean(episode_lengths)
        food_rate = foods_found / 5
        death_rate = quick_deaths / 5
        
        return avg_score, avg_length, food_rate, death_rate

def main():
    print("🚀 TRAINING WITH FIXED REWARDS!")
    print("=" * 45)
    print("💀 Death penalty: -35 (was -10)")
    print("🍎 Food reward: +15 (was +10)")
    print("🎯 Goal: Stop reward hacking!")
    print("=" * 45)
    
    # Create environment
    env = SnakeEnvironment()
    env = Monitor(env, "logs_fixed/")
    
    print(f"✅ Environment ready with improved rewards!")
    
    # Create fresh model (no loading existing one to avoid bad habits)
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
    
    # Test baseline (should be random)
    print(f"\n📊 BASELINE (Random AI):")
    test_ai_performance(model, env, "BASELINE")
    
    # Train with smart monitoring
    print(f"\n🏋️ TRAINING WITH SMART MONITORING:")
    print(f"Training for 40,000 steps...")
    print(f"Smart checks every 3,000 steps")
    
    smart_callback = SmartProgressCallback(eval_freq=3000)
    
    model.learn(
        total_timesteps=40000,
        callback=smart_callback,
        progress_bar=True
    )
    
    print(f"✅ Training completed!")
    
    # Save the model
    print(f"\n💾 Saving fixed-reward AI...")
    model.save("models/fixed_reward_ai")
    print(f"✅ Saved as 'fixed_reward_ai'")
    
    # Final comprehensive test
    print(f"\n📊 FINAL AI PERFORMANCE:")
    test_ai_performance(model, env, "FINAL")
    
    # Show progress summary
    if len(smart_callback.progress_data) > 1:
        show_smart_progress(smart_callback.progress_data)
    
    print(f"\n🎉 FIXED REWARD TRAINING COMPLETE!")

def test_ai_performance(model, env, phase):
    """Test AI performance with detailed analysis"""
    scores = []
    episode_lengths = []
    foods_found = 0
    quick_deaths = 0
    
    print(f"Testing {phase} performance (8 episodes)...")
    
    for episode in range(8):
        obs, info = env.reset()
        episode_steps = 0
        episode_score = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                episode_score = info['score']
                break
        
        scores.append(episode_score)
        episode_lengths.append(episode_steps)
        
        if episode_score > 0:
            foods_found += 1
        if episode_steps < 30:
            quick_deaths += 1
        
        if episode < 4:  # Show first 4 episodes
            death_type = " (QUICK!)" if episode_steps < 30 else ""
            print(f"   Episode {episode+1}: Score={episode_score}, Steps={episode_steps}{death_type}")
    
    avg_score = np.mean(scores)
    avg_length = np.mean(episode_lengths)
    food_rate = foods_found / 8
    death_rate = quick_deaths / 8
    
    print(f"📈 {phase} RESULTS:")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Average Episode Length: {avg_length:.1f}")
    print(f"   Food Success Rate: {food_rate:.1%}")
    print(f"   Total Foods Found: {foods_found}")
    print(f"   Quick Death Rate: {death_rate:.1%}")
    
    # Analysis
    if death_rate > 0.5:
        print(f"   ⚠️  CONCERN: High quick death rate - still reward hacking?")
    elif food_rate > 0.3:
        print(f"   🏆 EXCELLENT: High food success rate!")
    elif avg_length > 60:
        print(f"   📈 GOOD: Long episodes suggest exploration!")
    elif avg_score > 0:
        print(f"   🎯 PROMISING: Finding some food!")
    else:
        print(f"   😰 STRUGGLING: Needs more work")

def show_smart_progress(progress_data):
    """Show intelligent progress analysis"""
    print(f"\n📈 SMART TRAINING ANALYSIS:")
    print(f"   Steps → Score | Length | Food% | QuickDeath%")
    
    for data in progress_data:
        print(f"   {data['steps']:>6,} → {data['avg_score']:>4.2f} | {data['avg_episode_length']:>6.1f} | {data['food_success_rate']:>4.0%} | {data['quick_death_rate']:>9.0%}")
    
    # Trend analysis
    if len(progress_data) >= 2:
        first = progress_data[0]
        last = progress_data[-1]
        
        score_change = last['avg_score'] - first['avg_score']
        death_change = last['quick_death_rate'] - first['quick_death_rate']
        
        print(f"\n   📊 OVERALL TRENDS:")
        print(f"   Score change: {score_change:+.2f}")
        print(f"   Quick death change: {death_change:+.1%}")
        
        if score_change > 0.1 and death_change < 0.2:
            print(f"   🎉 SUCCESS: Improving without reward hacking!")
        elif death_change > 0.3:
            print(f"   ⚠️  WARNING: Learning to die quickly!")
        elif score_change > 0:
            print(f"   📈 PROGRESS: Slow but steady improvement!")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_fixed", exist_ok=True)
    
    main()