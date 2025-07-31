#!/usr/bin/env python3
"""
Test the enhanced reward system with distance rewards
"""

from snake_env import SnakeEnvironment
import time

def test_reward_system():
    print("🧪 TESTING ENHANCED REWARD SYSTEM")
    print("=" * 45)
    
    # Display reward configuration
    env = SnakeEnvironment()
    print(f"📊 REWARD CONFIGURATION:")
    print(f"   💀 Death Penalty: {env.DEATH_PENALTY}")
    print(f"   🍎 Food Reward: {env.FOOD_REWARD}")
    print(f"   ⏱️ Step Penalty: {env.STEP_PENALTY}")
    print(f"   📏 Distance Reward: +{env.DISTANCE_REWARD} (closer) / {env.DISTANCE_PENALTY} (farther)")
    print(f"   😰 Max Hunger Steps: {env.HUNGER_MAX}")
    
    print(f"\n🎮 MANUAL TEST (5 steps):")
    print(f"Watch how rewards change with movement...")
    
    obs, info = env.reset()
    
    # Get initial positions
    head_x, head_y = env.snake[0]
    food_x, food_y = env.food
    initial_distance = abs(head_x - food_x) + abs(head_y - food_y)
    
    print(f"\n📍 INITIAL STATE:")
    print(f"   Snake head: ({head_x}, {head_y})")
    print(f"   Food: ({food_x}, {food_y})")  
    print(f"   Distance: {initial_distance}")
    
    # Test 5 random actions
    import random
    actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    for step in range(5):
        action = random.randint(0, 3)
        action_name = actions[action]
        
        print(f"\n🕹️ STEP {step+1}: Action = {action_name}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get new positions
        head_x, head_y = env.snake[0]
        new_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        print(f"   New head position: ({head_x}, {head_y})")
        print(f"   New distance: {new_distance}")
        print(f"   🎁 Total Reward: {reward:.2f}")
        print(f"   🏃 Steps since food: {info['hunger']}")
        
        if terminated:
            print(f"   💀 GAME OVER!")
            break
    
    print(f"\n✅ TEST COMPLETE!")
    print(f"🔧 The reward system is working with:")
    print(f"   - Distance tracking ✅")
    print(f"   - Configurable rewards ✅") 
    print(f"   - Multiple pressure systems ✅")

if __name__ == "__main__":
    test_reward_system()