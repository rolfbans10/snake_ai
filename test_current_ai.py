#!/usr/bin/env python3
"""
Test the current AI model to see what it learned
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
import time

def test_ai():
    print("🧪 TESTING CURRENT AI MODEL")
    print("=" * 40)
    
    # Load the saved model
    try:
        model = PPO.load("models/best_model", device="cpu")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create environment
    env = SnakeEnvironment()
    
    print("\n🎮 TESTING AI PERFORMANCE (with time limits)...")
    
    total_scores = []
    total_steps = []
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        episode_score = 0
        episode_steps = 0
        max_steps = 200  # Prevent infinite episodes
        
        start_time = time.time()
        
        while episode_steps < max_steps:
            # AI chooses action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            # Print progress every 50 steps
            if episode_steps % 50 == 0:
                print(f"  Step {episode_steps}: Score = {info['score']}, Still alive...")
            
            if terminated or truncated:
                episode_score = info['score']
                print(f"  🎯 Game over at step {episode_steps}!")
                break
        
        if episode_steps >= max_steps:
            episode_score = info['score']
            print(f"  ⏰ Episode ended due to step limit ({max_steps})")
        
        episode_time = time.time() - start_time
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        
        print(f"  📊 Episode Results:")
        print(f"     Score: {episode_score}")
        print(f"     Steps: {episode_steps}")
        print(f"     Time: {episode_time:.1f} seconds")
    
    # Summary
    avg_score = sum(total_scores) / len(total_scores)
    avg_steps = sum(total_steps) / len(total_steps)
    
    print(f"\n📈 AI PERFORMANCE SUMMARY:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Best Score: {max(total_scores)}")
    print(f"   Survival Rate: {sum(1 for s in total_steps if s >= 100) / len(total_steps) * 100:.0f}% (lived 100+ steps)")
    
    # Compare to random baseline
    print(f"\n🎲 COMPARISON TO RANDOM:")
    print(f"   Random Agent: ~25 steps, 0 score")
    print(f"   Your AI: ~{avg_steps:.0f} steps, {avg_score:.1f} score")
    
    if avg_steps > 50:
        print(f"   🎉 SUCCESS: AI learned to survive much longer!")
    if avg_score > 0:
        print(f"   🍎 AMAZING: AI actually found food!")
    else:
        print(f"   🤔 Still learning: AI survives but hasn't found food efficiently yet")

if __name__ == "__main__":
    test_ai()