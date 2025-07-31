#!/usr/bin/env python3
"""
Extended training for even better hungry AI performance
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import matplotlib.pyplot as plt
import numpy as np

class ProgressCallback(BaseCallback):
    """Custom callback to track and display training progress"""
    
    def __init__(self, eval_freq=5000):
        super().__init__()
        self.eval_freq = eval_freq
        self.scores_history = []
        self.steps_history = []
        self.food_rate_history = []
        
    def _on_step(self) -> bool:
        # Evaluate every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            print(f"\n📊 PROGRESS CHECK at {self.n_calls} steps:")
            scores, steps, food_rate = self.evaluate_current_ai()
            
            self.scores_history.append(scores)
            self.steps_history.append(steps)
            self.food_rate_history.append(food_rate)
            
            print(f"   Average Score: {scores:.2f}")
            print(f"   Average Steps: {steps:.1f}")
            print(f"   Food Success Rate: {food_rate:.1%}")
            
            # Show improvement trend
            if len(self.scores_history) > 1:
                score_change = scores - self.scores_history[-2]
                print(f"   Score Change: {score_change:+.2f}")
                
        return True
    
    def evaluate_current_ai(self):
        """Quick evaluation of current AI"""
        env = SnakeEnvironment()
        
        total_scores = []
        total_steps = []
        foods_found = 0
        
        for _ in range(5):  # Quick 5-episode test
            obs, info = env.reset()
            episode_score = 0
            episode_steps = 0
            
            while episode_steps < 200:  # Limit episode length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                if terminated or truncated:
                    episode_score = info['score']
                    break
            
            total_scores.append(episode_score)
            total_steps.append(episode_steps)
            if episode_score > 0:
                foods_found += 1
        
        avg_score = np.mean(total_scores)
        avg_steps = np.mean(total_steps)
        food_rate = foods_found / 5
        
        return avg_score, avg_steps, food_rate

def main():
    print("🚀 EXTENDED TRAINING FOR SUPERIOR AI!")
    print("=" * 50)
    
    # Create environment
    class LongTrainingEnv(SnakeEnvironment):
        def __init__(self):
            super().__init__()
            self.max_steps = 250  # Slightly longer episodes
            self.step_count = 0
            
        def reset(self, seed=None):
            self.step_count = 0
            return super().reset(seed)
            
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.step_count += 1
            
            if self.step_count >= self.max_steps:
                truncated = True
                reward -= 1  # Penalty for taking too long
                
            return obs, reward, terminated, truncated, info
    
    env = LongTrainingEnv()
    env = Monitor(env, "logs_extended/")
    
    print(f"✅ Extended training environment ready!")
    
    # Load existing model to continue training
    try:
        model = PPO.load("models/hungry_snake_ai", env=env, device="cpu")
        print(f"✅ Loaded existing hungry AI to continue training!")
        print(f"   Starting from previous progress...")
    except:
        print(f"❌ No existing model found, creating new one...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            device="cpu",
        )
    
    # Test current performance
    print(f"\n📊 CURRENT AI PERFORMANCE:")
    test_ai_performance(model, env, "CURRENT")
    
    # Extended training with progress tracking
    print(f"\n🏋️ EXTENDED TRAINING SESSION:")
    print(f"Training for 75,000 additional steps...")
    print(f"Progress checks every 5,000 steps")
    
    # Create progress callback
    progress_callback = ProgressCallback(eval_freq=5000)
    
    # Train with progress monitoring
    model.learn(
        total_timesteps=75000,  # Much longer training!
        callback=progress_callback,
        progress_bar=True,
        reset_num_timesteps=False  # Continue from where we left off
    )
    
    print(f"✅ Extended training completed!")
    
    # Save the improved model
    print(f"\n💾 Saving super-trained AI...")
    model.save("models/super_hungry_ai")
    print(f"✅ Saved as 'super_hungry_ai'")
    
    # Final comprehensive test
    print(f"\n📊 FINAL AI PERFORMANCE:")
    test_ai_performance(model, env, "FINAL")
    
    # Show training progress graph
    if len(progress_callback.scores_history) > 1:
        show_training_progress(progress_callback)
    
    print(f"\n🎉 EXTENDED TRAINING COMPLETE!")
    print(f"Your AI should now be significantly better!")

def test_ai_performance(model, env, phase):
    """Comprehensive AI performance test"""
    total_scores = []
    total_steps = []
    total_foods = 0
    
    print(f"Testing {phase} performance (10 episodes)...")
    
    for episode in range(10):  # More episodes for better statistics
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
        total_foods += episode_score
        
        if episode < 5:  # Show first 5 episodes
            print(f"   Episode {episode+1}: Score={episode_score}, Steps={episode_steps}")
    
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    food_success_rate = sum(1 for s in total_scores if s > 0) / len(total_scores)
    
    print(f"📈 {phase} RESULTS:")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Food Success Rate: {food_success_rate:.1%}")
    print(f"   Total Foods Found: {total_foods}")
    
    if avg_score > 0.5:
        print(f"   🏆 EXCELLENT: Consistently finding food!")
    elif avg_score > 0.2:
        print(f"   🎯 GOOD: Regular food finding!")
    elif avg_score > 0:
        print(f"   📈 IMPROVING: Occasional food finding!")
    else:
        print(f"   😰 STRUGGLING: Needs more training")

def show_training_progress(callback):
    """Show a simple progress chart"""
    print(f"\n📈 TRAINING PROGRESS SUMMARY:")
    
    steps = [i * 5000 for i in range(len(callback.scores_history))]
    scores = callback.scores_history
    
    print(f"   Training Steps → Average Score")
    for i, (step, score) in enumerate(zip(steps, scores)):
        stars = "★" * int(score * 10)  # Visual representation
        print(f"   {step:>6,} steps → {score:.2f} {stars}")
    
    if len(scores) > 1:
        improvement = scores[-1] - scores[0]
        print(f"\n   📊 Total Improvement: {improvement:+.2f} points")
        
        if improvement > 0.1:
            print(f"   🚀 SIGNIFICANT IMPROVEMENT!")
        elif improvement > 0:
            print(f"   📈 Good progress!")
        else:
            print(f"   🤔 May need more training or different approach")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_extended", exist_ok=True)
    
    main()