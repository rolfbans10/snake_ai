#!/usr/bin/env python3
"""
Watch Simple AI Play Snake
Load and visualize the trained simple snake AI
"""

from snake_env_simple import SimpleSnakeEnvironment
from stable_baselines3 import PPO
import time
import os

def main():
    print("🎮 WATCHING SIMPLE AI PLAY SNAKE")
    print("=" * 40)
    
    # Check if model exists
    model_path = "models/simple_snake_ai"
    if not os.path.exists(f"{model_path}.zip"):
        print("❌ No trained model found!")
        print(f"   Looking for: {model_path}.zip")
        print("   Please run 'python train_simple_ai.py' first to train the AI!")
        return
    
    # Load the trained model
    print("🧠 Loading trained AI...")
    model = PPO.load(model_path)
    print("✅ AI loaded successfully!")
    
    # Create environment for visualization
    env = SimpleSnakeEnvironment()
    
    print("\n🐍 Starting AI gameplay...")
    print("   Close the game window to stop watching")
    print("   Press Ctrl+C to stop from terminal")
    
    try:
        games_played = 0
        total_score = 0
        
        while True:
            games_played += 1
            print(f"\n🎯 Game {games_played}")
            
            # Reset environment
            obs, info = env.reset()
            episode_steps = 0
            episode_reward = info['reward_balance']
            
            while True:
                # Render the game
                env.render()
                
                # AI makes decision
                action, _ = model.predict(obs, deterministic=True)
                
                # Take action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                # Add small delay to make it watchable
                time.sleep(0.1)  # 100ms delay
                
                # Check if game is over
                if terminated or truncated:
                    final_score = info['score']
                    final_balance = info['reward_balance']
                    total_score += final_score
                    
                    print(f"   Game Over! Score: {final_score}, Steps: {episode_steps}")
                    print(f"   Final Balance: {final_balance:.1f}")
                    print(f"   Average Score: {total_score/games_played:.1f}")
                    
                    # Wait a bit before next game
                    time.sleep(2)
                    break
                
                # Prevent infinite games
                if episode_steps > 2000:
                    print(f"   Long game ended: Score {info['score']}, Steps {episode_steps}")
                    time.sleep(1)
                    break
    
    except KeyboardInterrupt:
        print(f"\n👋 Stopped watching. Games played: {games_played}")
        if games_played > 0:
            print(f"   Average score: {total_score/games_played:.1f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        try:
            import pygame
            pygame.quit()
        except:
            pass

def test_ai_performance(episodes=20):
    """Test AI performance without visualization"""
    print(f"\n📊 Testing AI performance ({episodes} episodes)...")
    
    model_path = "models/simple_snake_ai"
    if not os.path.exists(f"{model_path}.zip"):
        print("❌ No trained model found!")
        return
    
    model = PPO.load(model_path)
    env = SimpleSnakeEnvironment()
    
    scores = []
    steps = []
    balances = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated or episode_steps > 2000:
                scores.append(info['score'])
                steps.append(episode_steps)
                balances.append(info['reward_balance'])
                break
        
        if (episode + 1) % 5 == 0:
            print(f"   Completed {episode + 1}/{episodes} episodes...")
    
    # Print results
    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps) / len(steps)
    avg_balance = sum(balances) / len(balances)
    max_score = max(scores)
    
    print(f"\n📈 Performance Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Best Score: {max_score}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Balance: {avg_balance:.1f}")
    
    # Performance rating
    if avg_score >= 5:
        print("🏆 EXCELLENT performance!")
    elif avg_score >= 3:
        print("🎯 GOOD performance!")
    elif avg_score >= 1:
        print("📈 FAIR performance - room for improvement")
    else:
        print("🔄 POOR performance - needs more training")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_ai_performance()
    else:
        main()