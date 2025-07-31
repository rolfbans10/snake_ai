#!/usr/bin/env python3
"""
Watch your trained AI play Snake in real-time!
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
import time

def watch_ai_play():
    print("🎮 WATCH YOUR AI PLAY SNAKE!")
    print("=" * 40)
    
    # Load the trained model
    try:
        model = PPO.load("models/snake_ai_simple", device="cpu")
        print("✅ AI loaded successfully!")
    except:
        try:
            model = PPO.load("models/best_model", device="cpu")
            print("✅ AI loaded successfully!")
        except:
            print("❌ No trained model found!")
            return
    
    # Create environment for visual play
    env = SnakeEnvironment()
    
    print("\n🐍 Instructions:")
    print("   - Watch your AI play Snake!")
    print("   - Close the window to stop")
    print("   - Your AI will automatically restart when it dies")
    
    episode = 1
    
    while True:
        print(f"\n🎯 Episode {episode}")
        obs, info = env.reset()
        step_count = 0
        
        try:
            while True:
                # AI chooses action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Render the game
                env.render()
                
                step_count += 1
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"   Step {step_count}: Score = {info['score']}, Still alive!")
                
                # Slow down so we can watch
                time.sleep(0.1)  # 10 FPS
                
                if terminated or truncated:
                    final_score = info['score']
                    print(f"   🏁 Episode {episode} finished!")
                    print(f"      Final Score: {final_score}")
                    print(f"      Steps Survived: {step_count}")
                    
                    # Show end screen for a moment
                    for _ in range(20):  # Show "Game Over" for 2 seconds
                        env.render()
                        time.sleep(0.1)
                    
                    break
                    
                # Prevent super long episodes
                if step_count > 500:
                    print(f"   ⏰ Episode ended due to step limit")
                    break
        
        except KeyboardInterrupt:
            print("\n👋 Stopping AI demo...")
            break
        except:
            print("\n🚪 Window closed, stopping demo...")
            break
        
        episode += 1
    
    env.close()
    print("Thanks for watching your AI play! 🎉")

if __name__ == "__main__":
    watch_ai_play()