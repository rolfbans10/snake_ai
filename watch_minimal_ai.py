#!/usr/bin/env python3
"""
Watch Minimal AI Play Snake
Load and visualize trained minimal snake AI models.

Usage:
    python watch_minimal_ai.py                              # Default model
    python watch_minimal_ai.py --model tiny_10x10           # Specific model
    python watch_minimal_ai.py --model minimal_64x64 --test # Test mode
    python watch_minimal_ai.py --list                       # List available models
"""

from snake_env_simple_minimal import MinimalSnakeEnvironment
from stable_baselines3 import PPO
import argparse
import time
import os
import glob


def list_available_models():
    """List all available minimal AI models"""
    print("📁 Available models in models/ directory:")
    print("=" * 50)
    
    model_files = glob.glob("models/*.zip")
    
    if not model_files:
        print("   No models found!")
        print("   Train some with: python train_minimal_ai.py")
        return []
    
    models = []
    for f in sorted(model_files):
        name = os.path.basename(f).replace(".zip", "")
        size = os.path.getsize(f) / 1024  # KB
        models.append(name)
        print(f"   - {name} ({size:.1f} KB)")
    
    print(f"\nTotal: {len(models)} models")
    return models


def watch_ai_play(model_name: str, speed: float = 0.1):
    """Watch AI play with visualization"""
    print("🎮 WATCHING MINIMAL AI PLAY SNAKE")
    print("=" * 50)
    
    # Determine model path
    if model_name.startswith("models/"):
        model_path = model_name.replace(".zip", "")
    else:
        model_path = f"models/{model_name}"
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Model not found: {model_path}.zip")
        print("\nAvailable models:")
        list_available_models()
        return
    
    # Load the trained model
    print(f"🧠 Loading model: {model_path}")
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    
    # Create environment for visualization
    env = MinimalSnakeEnvironment()
    
    print(f"\n🐍 Starting AI gameplay...")
    print(f"   Model: {model_name}")
    print(f"   Speed: {speed}s per frame")
    print(f"   Close the game window to stop")
    print(f"   Press Ctrl+C to stop from terminal")
    
    try:
        games_played = 0
        total_score = 0
        
        while True:
            games_played += 1
            print(f"\n🎯 Game {games_played}")
            
            # Reset environment
            obs, info = env.reset()
            episode_steps = 0
            
            while True:
                # Render the game
                env.render()
                
                # AI makes decision
                action, _ = model.predict(obs, deterministic=True)
                
                # Take action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                # Add delay to make it watchable
                time.sleep(speed)
                
                # Check if game is over
                if terminated or truncated:
                    final_score = info['score']
                    total_score += final_score
                    
                    print(f"   Game Over! Score: {final_score}, Steps: {episode_steps}")
                    print(f"   Average Score: {total_score/games_played:.1f}")
                    
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
        try:
            import pygame
            pygame.quit()
        except:
            pass


def test_ai_performance(model_name: str, episodes: int = 20):
    """Test AI performance without visualization"""
    print(f"📊 Testing AI Performance")
    print("=" * 50)
    
    # Determine model path
    if model_name.startswith("models/"):
        model_path = model_name.replace(".zip", "")
    else:
        model_path = f"models/{model_name}"
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Model not found: {model_path}.zip")
        print("\nAvailable models:")
        list_available_models()
        return
    
    print(f"🧠 Model: {model_name}")
    print(f"🎮 Episodes: {episodes}")
    
    model = PPO.load(model_path)
    env = MinimalSnakeEnvironment()
    
    scores = []
    steps = []
    balances = []
    
    print(f"\nRunning {episodes} episodes...")
    
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
    min_score = min(scores)
    
    print(f"\n📈 Results for: {model_name}")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Best Score: {max_score}")
    print(f"   Worst Score: {min_score}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Balance: {avg_balance:.1f}")
    
    # Performance rating
    if avg_score >= 10:
        print("\n🏆 OUTSTANDING!")
    elif avg_score >= 5:
        print("\n🎯 EXCELLENT!")
    elif avg_score >= 3:
        print("\n👍 GREAT!")
    elif avg_score >= 1:
        print("\n📈 GOOD - room for improvement")
    else:
        print("\n🔄 NEEDS MORE TRAINING")
    
    return {
        "model": model_name,
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_steps": avg_steps
    }


def compare_models(model_names: list, episodes: int = 20):
    """Compare multiple models"""
    print("🔬 MODEL COMPARISON")
    print("=" * 60)
    
    results = []
    
    for model_name in model_names:
        print(f"\n--- Testing: {model_name} ---")
        result = test_ai_performance(model_name, episodes)
        if result:
            results.append(result)
    
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("📊 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<30} {'Avg Score':>10} {'Max Score':>10} {'Avg Steps':>10}")
        print("-" * 60)
        
        # Sort by average score
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        
        for i, r in enumerate(results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{medal} {r['model']:<28} {r['avg_score']:>10.2f} {r['max_score']:>10} {r['avg_steps']:>10.1f}")
        
        print("\n🏆 Winner: " + results[0]['model'])


def show_observation_details():
    """Show what the minimal observation looks like"""
    print("🔍 MINIMAL OBSERVATION BREAKDOWN")
    print("=" * 50)
    
    env = MinimalSnakeEnvironment()
    obs, info = env.reset()
    
    print(f"Total dimensions: {len(obs)} (vs 777 in full board version)")
    print(f"\nFeature breakdown:")
    print(f"  [0-3]   Danger indicators: {obs[0:4]}")
    print(f"  [4-5]   Food direction: {obs[4:6]}")
    print(f"  [6-9]   Wall distances: {obs[6:10]}")
    print(f"  [10]    Snake length: {obs[10]}")
    print(f"  [11]    Distance to food: {obs[11]}")
    print(f"  [12]    Current direction: {obs[12]}")
    print(f"  [13]    Body proximity: {obs[13]}")
    print(f"  [14]    Reward balance: {obs[14]}")
    print(f"  [15]    Safe moves count: {obs[15]}")
    print(f"  [16-17] Movement history: {obs[16:18]}")
    print(f"  [18-19] Tail direction: {obs[18:20]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🐍 Watch Minimal Snake AI Play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch_minimal_ai.py                                # Watch default model
  python watch_minimal_ai.py --model tiny_10x10             # Watch specific model
  python watch_minimal_ai.py --model minimal_64x64 --test   # Test specific model
  python watch_minimal_ai.py --list                         # List available models
  python watch_minimal_ai.py --compare tiny_10x10 medium_64x64  # Compare models
  python watch_minimal_ai.py --speed 0.05                   # Faster playback
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="minimal_snake_ai",
        help="Model name to load (default: minimal_snake_ai)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test model performance without visualization"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes for testing (default: 20)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Compare multiple models (space-separated names)"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Playback speed in seconds per frame (default: 0.1)"
    )
    
    parser.add_argument(
        "--show-obs",
        action="store_true",
        help="Show observation feature breakdown"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.show_obs:
        show_observation_details()
    elif args.compare:
        compare_models(args.compare, args.episodes)
    elif args.test:
        test_ai_performance(args.model, args.episodes)
    else:
        watch_ai_play(args.model, args.speed)
