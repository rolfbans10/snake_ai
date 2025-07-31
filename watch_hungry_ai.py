#!/usr/bin/env python3
"""
Watch your HUNGRY AI play - no more circling!
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
import time

def watch_hungry_ai():
    print("🍽️  WATCH YOUR HUNGRY AI HUNT FOR FOOD!")
    print("=" * 50)
    
    # Load the hungry AI
    try:
        model = PPO.load("models/hungry_snake_ai", device="cpu")
        print("✅ Hungry AI loaded successfully!")
    except:
        print("❌ No hungry AI model found! Run train_hungry_ai.py first")
        return
    
    # Create environment 
    env = SnakeEnvironment()
    
    print("\n🧠 Your AI now has HUNGER pressure!")
    print("   - Watch how it actively seeks food")
    print("   - No more safe circling!")
    print("   - Urgency increases over time")
    print("   - Close window to stop")
    
    episode = 1
    
    while True:
        print(f"\n🎯 Episode {episode}")
        obs, info = env.reset()
        step_count = 0
        foods_found = 0
        
        try:
            while True:
                # AI chooses action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Render the game
                env.render()
                
                step_count += 1
                
                # Track food finding
                if info['score'] > foods_found:
                    foods_found = info['score']
                    print(f"   🍎 FOOD #{foods_found} FOUND at step {step_count}! Hunger: {info['hunger']}")
                
                # Show hunger status
                if step_count % 25 == 0:
                    hunger_level = info['hunger']
                    if hunger_level < 25:
                        status = "😊 Content"
                    elif hunger_level < 50:
                        status = "😐 Getting hungry"
                    elif hunger_level < 100:
                        status = "😰 Very hungry"
                    else:
                        status = "🔥 STARVING!"
                    
                    print(f"   Step {step_count}: {status} (Hunger: {hunger_level})")
                
                # Slow down so we can watch
                time.sleep(0.15)  # Slightly faster than before
                
                if terminated or truncated:
                    final_score = info['score']
                    final_hunger = info['hunger']
                    print(f"   🏁 Episode {episode} finished!")
                    print(f"      Final Score: {final_score}")
                    print(f"      Steps Survived: {step_count}")
                    print(f"      Max Hunger Reached: {final_hunger}")
                    
                    if final_score > 0:
                        print(f"      🎉 SUCCESS: Found {final_score} food(s)!")
                    else:
                        print(f"      💀 Died of starvation/collision")
                    
                    # Show end screen briefly
                    for _ in range(15):
                        env.render()
                        time.sleep(0.1)
                    
                    break
                    
                # Prevent super long episodes
                if step_count > 300:
                    print(f"   ⏰ Episode ended (300 step limit)")
                    break
        
        except KeyboardInterrupt:
            print("\n👋 Stopping hungry AI demo...")
            break
        except:
            print("\n🚪 Window closed, stopping demo...")
            break
        
        episode += 1
    
    env.close()
    print("Thanks for watching your HUNGRY AI! 🍽️🧠")

if __name__ == "__main__":
    watch_hungry_ai()