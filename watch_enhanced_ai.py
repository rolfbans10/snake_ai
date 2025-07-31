#!/usr/bin/env python3
"""
Watch the Enhanced AI in Action!
Shows the AI trained with multi-layered reward system
"""

from snake_env import SnakeEnvironment  
from stable_baselines3 import PPO
import pygame
import time

def main():
    print("🎮 WATCH ENHANCED AI PLAY!")
    print("=" * 40)
    
    # Load the enhanced AI
    try:
        model = PPO.load("models/enhanced_reward_ai")
        print("✅ Enhanced AI loaded successfully!")
    except:
        print("❌ No enhanced AI found. Train first with: python train_enhanced_ai.py")
        return
    
    # Create environment  
    env = SnakeEnvironment()
    
    print(f"🎯 ENHANCED REWARD SYSTEM:")
    print(f"   💀 Death: {env.DEATH_PENALTY}")
    print(f"   🍎 Food: +{env.FOOD_REWARD}")  
    print(f"   📏 Distance: ±{env.DISTANCE_REWARD}")
    print(f"   ⏱️ Step: {env.STEP_PENALTY}")
    print(f"   😰 Hunger: Progressive penalty")
    print("=" * 40)
    
    print("🧠 Your AI has multiple guidance systems:")
    print("   - Distance rewards (move toward food)")
    print("   - Hunger pressure (find food quickly)")
    print("   - Death avoidance (don't crash)")
    print("   - Efficiency pressure (don't waste steps)")
    print("   - Close window to stop")
    
    episode = 1
    
    while True:
        print(f"\n🎯 Episode {episode}")
        obs, info = env.reset()
        
        episode_steps = 0
        foods_found = 0
        max_hunger = 0
        
        while True:
            # Render the game
            env.render()
            
            # Let AI decide next move
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_steps += 1
            
            # Track progress
            if info['score'] > foods_found:
                foods_found = info['score']
                print(f"   🍎 FOOD #{foods_found} FOUND at step {episode_steps}! Hunger: {info['hunger']}")
            
            # Track hunger
            max_hunger = max(max_hunger, info['hunger'])
            
            # Show periodic updates
            if episode_steps % 25 == 0:
                if info['hunger'] < 25:
                    print(f"   Step {episode_steps}: 😊 Content (Hunger: {info['hunger']})")
                elif info['hunger'] < 50:
                    print(f"   Step {episode_steps}: 😐 Getting hungry (Hunger: {info['hunger']})")
                elif info['hunger'] < 100:
                    print(f"   Step {episode_steps}: 😰 Very hungry (Hunger: {info['hunger']})")
                else:
                    print(f"   Step {episode_steps}: 🔥 STARVING! (Hunger: {info['hunger']})")
            
            # Check for pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("🚪 Window closed, stopping demo...")
                    pygame.quit()
                    print("Thanks for watching your Enhanced AI! 🤖✨")
                    return
            
            # Slow down for viewing
            time.sleep(0.05)  # 20 FPS for comfortable viewing
            
            if terminated or truncated:
                break
        
        print(f"   🏁 Episode {episode} finished!")
        print(f"      Final Score: {foods_found}")
        print(f"      Steps Survived: {episode_steps}")
        print(f"      Max Hunger Reached: {max_hunger}")
        
        if foods_found > 0:
            print(f"      🎉 SUCCESS: Found {foods_found} food(s)!")
        elif episode_steps > 50:
            print(f"      📈 PROGRESS: Survived {episode_steps} steps!")
        else:
            print(f"      🔧 LEARNING: Quick death in {episode_steps} steps")
        
        episode += 1

if __name__ == "__main__":
    main()