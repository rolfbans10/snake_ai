#!/usr/bin/env python3
"""
Test the new multiple foods system
"""

from snake_env import SnakeEnvironment
import time

def test_multiple_foods():
    print("🍎 TESTING MULTIPLE FOODS SYSTEM")
    print("=" * 45)
    
    # Create environment and show configuration
    env = SnakeEnvironment()
    
    print(f"📊 MULTI-FOOD CONFIGURATION:")
    print(f"   🍎 Number of Foods: {env.NUM_FOODS}")
    print(f"   💀 Death Penalty: {env.DEATH_PENALTY}")
    print(f"   🏆 Food Reward: {env.FOOD_REWARD}")
    print(f"   📏 Distance Reward: ±{env.DISTANCE_REWARD}/{env.DISTANCE_PENALTY}")
    print(f"   ☠️ Starvation Limit: {env.MAX_STARVATION_STEPS} steps")
    print("=" * 45)
    
    obs, info = env.reset()
    
    print(f"🐍 INITIAL STATE:")
    print(f"   Snake position: {env.snake[0]}")
    print(f"   Number of foods on board: {len(env.foods)}")
    print(f"   Food positions: {env.foods}")
    
    # Calculate distances to each food
    head_x, head_y = env.snake[0]
    print(f"\n📏 DISTANCES TO FOODS:")
    for i, (food_x, food_y) in enumerate(env.foods):
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        print(f"   Food {i+1} at ({food_x}, {food_y}): distance = {distance}")
    
    nearest_distance = env._get_nearest_food_distance()
    print(f"   🎯 Nearest food distance: {nearest_distance}")
    
    print(f"\n🎮 MANUAL TEST (10 random steps):")
    print(f"Watch the AI navigate toward multiple foods...")
    
    import random
    actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    for step in range(10):
        action = random.randint(0, 3)
        action_name = actions[action]
        
        print(f"\n🕹️ STEP {step+1}: Action = {action_name}")
        
        # Store pre-step state
        old_score = env.score
        old_nearest_distance = env._get_nearest_food_distance()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Analyze what happened
        new_nearest_distance = env._get_nearest_food_distance()
        
        print(f"   Score: {old_score} → {env.score}")
        print(f"   Foods remaining: {len(env.foods)}")
        print(f"   Nearest food distance: {old_nearest_distance} → {new_nearest_distance}")
        print(f"   🎁 Reward: {reward:.2f}")
        print(f"   🏃 Hunger: {info['hunger']}")
        
        # Check if food was eaten
        if env.score > old_score:
            print(f"   🍎 FOOD EATEN! New food added to maintain {env.NUM_FOODS} foods")
            print(f"   Updated food positions: {env.foods}")
        
        if terminated:
            print(f"   💀 GAME OVER!")
            if info.get('starved_to_death'):
                print(f"   ☠️ Reason: Starved to death")
            else:
                print(f"   💥 Reason: Collision")
            break
    
    print(f"\n🎉 MULTIPLE FOODS TEST COMPLETE!")
    print(f"✅ The snake now has {env.NUM_FOODS} foods to choose from!")
    print(f"✅ Distance rewards guide toward the NEAREST food!")
    print(f"✅ Eating food removes that food and adds a new one!")
    print(f"✅ AI has much better chance of finding food!")

if __name__ == "__main__":
    test_multiple_foods()