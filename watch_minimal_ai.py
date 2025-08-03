#!/usr/bin/env python3
"""
Watch Minimal AI Play Snake
Load and visualize the trained minimal snake AI (15-feature version)
"""

from snake_env_simple_minimal import MinimalSnakeEnvironment
from stable_baselines3 import PPO
import time
import os

def main():
    print("🎮 WATCHING MINIMAL AI PLAY SNAKE")
    print("=" * 40)
    print("🚀 Using MINIMAL feature set (15 dims vs 777)")
    
    # Check if model exists
    model_path = "models/minimal_snake_ai"
    if not os.path.exists(f"{model_path}.zip"):
        print("❌ No trained model found!")
        print(f"   Looking for: {model_path}.zip")
        print("   Please run 'python train_minimal_ai.py' first to train the AI!")
        return
    
    # Load the trained model
    print("🧠 Loading trained minimal AI...")
    model = PPO.load(model_path)
    print("✅ AI loaded successfully!")
    print("   Model uses only 15 features - much more efficient!")
    
    # Create environment for visualization
    env = MinimalSnakeEnvironment()
    
    print("\n🐍 Starting AI gameplay...")
    print("   Close the game window to stop watching")
    print("   Press Ctrl+C to stop from terminal")
    print("   🎯 Watch for better performance with focused features!")
    
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
            
            # Show initial observation (for debugging)
            if games_played == 1:
                print(f"   📊 Initial observation (15 features): {obs}")
                print(f"   Features: [dangers(4), food_dir(2), walls(4), length(1), distance(1), direction(1), body_prox(1), balance(1)]")
            
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
    print(f"\n📊 Testing Minimal AI performance ({episodes} episodes)...")
    
    model_path = "models/minimal_snake_ai"
    if not os.path.exists(f"{model_path}.zip"):
        print("❌ No trained model found!")
        return
    
    model = PPO.load(model_path)
    env = MinimalSnakeEnvironment()
    
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
    
    print(f"\n📈 Minimal AI Performance Results:")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Best Score: {max_score}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Balance: {avg_balance:.1f}")
    
    # Performance rating
    if avg_score >= 10:
        print("🏆 OUTSTANDING performance! Minimal features work amazingly!")
    elif avg_score >= 5:
        print("🎯 EXCELLENT performance! Minimal AI is very effective!")
    elif avg_score >= 3:
        print("👍 GREAT performance! Minimal features are working well!")
    elif avg_score >= 1:
        print("📈 GOOD performance! Better than random, room for improvement")
    else:
        print("🔄 NEEDS IMPROVEMENT - consider more training")
    
    # Compare to expected baselines
    print(f"\n⚖️ PERFORMANCE COMPARISON:")
    print(f"   Random Agent: ~80 steps, 0 score")
    print(f"   Full Board AI (5k steps): ~24 steps, 0.2 score")
    print(f"   Minimal AI: {avg_steps:.0f} steps, {avg_score:.1f} score")
    
    if avg_score > 0.2:
        print(f"   🎉 MINIMAL AI WINS! Better than full board version!")
    if avg_steps > 50:
        print(f"   🎯 GREAT SURVIVAL! Much better than random!")

def show_observation_details():
    """Show what the minimal observation looks like"""
    print("🔍 MINIMAL OBSERVATION BREAKDOWN:")
    print("=" * 40)
    
    env = MinimalSnakeEnvironment()
    obs, info = env.reset()
    
    print(f"Total dimensions: {len(obs)} (vs 777 in full board version)")
    print(f"Observation values: {obs}")
    print(f"\nFeature breakdown:")
    print(f"  [0-3]   Danger indicators: {obs[0:4]} (up,right,down,left - 1=danger, 0=safe)")
    print(f"  [4-5]   Food direction: {obs[4:6]} (dx,dy normalized to food)")
    print(f"  [6-9]   Wall distances: {obs[6:10]} (up,right,down,left normalized)")
    print(f"  [10]    Snake length: {obs[10]} (current body size)")
    print(f"  [11]    Distance to food: {obs[11]} (normalized)")
    print(f"  [12]    Current direction: {obs[12]} (0=up,1=right,2=down,3=left)")
    print(f"  [13]    Body proximity: {obs[13]} (distance to nearest body segment)")
    print(f"  [14]    Reward balance: {obs[14]} (current performance)")
    print(f"\n🚀 Much cleaner and focused than 777-dimension board!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_ai_performance()
        elif sys.argv[1] == "--show-obs":
            show_observation_details()
        elif sys.argv[1] in ['-h', '--help']:
            print("🐍 Minimal Snake AI Viewer")
            print("Usage: python watch_minimal_ai.py [option]")
            print("")
            print("Options:")
            print("  (no args)    Watch AI play with visualization")
            print("  --test       Test AI performance (20 episodes, no graphics)")
            print("  --show-obs   Show observation breakdown")
            print("  --help       Show this help message")
            print("")
            print("Requirements:")
            print("  Trained model: models/minimal_snake_ai.zip")
            print("  Train with: python train_minimal_ai.py")
        else:
            print(f"❌ Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        main()