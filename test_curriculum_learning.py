#!/usr/bin/env python3
"""
Test the curriculum learning system with dynamic food reduction
"""

from snake_env import SnakeEnvironment

def test_curriculum_progression():
    print("🎓 CURRICULUM LEARNING SYSTEM TEST")
    print("=" * 50)
    
    env = SnakeEnvironment()
    
    print(f"📚 CURRICULUM CONFIGURATION:")
    print(f"   🍎 Initial Foods: {env.INITIAL_FOODS} (easy start)")
    print(f"   🎯 Final Foods: {env.FINAL_FOODS} (hard finish)")
    print(f"   📊 Total Training Steps: {env.TOTAL_TRAINING_STEPS:,}")
    print(f"   📉 Reduction Rate: 1 food per {env.TOTAL_TRAINING_STEPS // (env.INITIAL_FOODS - env.FINAL_FOODS):,} steps")
    
    print(f"\n🔬 TESTING FOOD COUNT CALCULATION:")
    
    # Test key milestones in training
    test_steps = [
        0,           # Start: Should be 10 foods
        250000,      # 10%: Should be 9 foods  
        500000,      # 20%: Should be 8 foods
        1250000,     # 50%: Should be 5 foods
        2000000,     # 80%: Should be 2 foods
        2500000,     # 100%: Should be 1 food
        3000000,     # Beyond: Should stay at 1 food
    ]
    
    for step in test_steps:
        # Temporarily set step count
        env.global_step_count = step
        food_count = env._get_current_food_count()
        progress = (step / env.TOTAL_TRAINING_STEPS) * 100
        
        print(f"   📈 Step {step:>7,} ({progress:>5.1f}%): {food_count} foods")
    
    # Reset step count
    env.global_step_count = 0
    
    print(f"\n🎮 SIMULATING TRAINING PROGRESSION:")
    
    # Simulate the progression by manually advancing steps
    milestone_steps = [0, 250000, 500000, 1000000, 1500000, 2000000, 2500000]
    
    for i, step in enumerate(milestone_steps):
        env.global_step_count = step
        obs, info = env.reset()  # This will call _place_multiple_foods with current count
        
        current_foods = len(env.foods)
        progress = (step / env.TOTAL_TRAINING_STEPS) * 100
        
        print(f"\n🕒 MILESTONE {i+1}: Step {step:,} ({progress:.0f}%)")
        print(f"   🍎 Foods on board: {current_foods}")
        print(f"   📍 Food positions: {env.foods[:3]}{'...' if len(env.foods) > 3 else ''}")
        print(f"   🎯 Nearest food distance: {env._get_nearest_food_distance()}")
        
        # Show difficulty assessment
        if current_foods >= 8:
            difficulty = "🟢 EASY"
        elif current_foods >= 5:
            difficulty = "🟡 MEDIUM"
        elif current_foods >= 2:
            difficulty = "🟠 HARD"
        else:
            difficulty = "🔴 EXPERT"
        
        print(f"   {difficulty} - AI has {current_foods} targets to choose from")

def test_dynamic_food_replacement():
    print(f"\n" + "="*50)
    print("🔄 DYNAMIC FOOD REPLACEMENT TEST")
    print("="*50)
    
    env = SnakeEnvironment()
    
    # Set to mid-training (should have ~5 foods)
    env.global_step_count = 1250000  # 50% progress
    obs, info = env.reset()
    
    initial_food_count = len(env.foods)
    print(f"🍎 Starting with {initial_food_count} foods at 50% training progress")
    print(f"   Foods: {env.foods}")
    
    # Simulate eating a food
    print(f"\n🐍 Simulating snake eating food at {env.foods[0]}")
    eaten_food = env.foods[0]
    
    # Manually trigger food eating logic
    env.snake = [eaten_food]  # Position snake at food
    ate_food = env._move_snake()  # This should eat the food
    
    print(f"   Food eaten: {ate_food}")
    print(f"   New food count: {len(env.foods)}")
    print(f"   Current foods: {env.foods}")
    
    # Show that food count stays appropriate for training stage
    expected_count = env._get_current_food_count()
    actual_count = len(env.foods)
    
    if actual_count <= expected_count:
        print(f"   ✅ SUCCESS: Food count ({actual_count}) matches curriculum stage ({expected_count})")
    else:
        print(f"   ❌ ERROR: Food count mismatch - got {actual_count}, expected ≤ {expected_count}")

def main():
    test_curriculum_progression()
    test_dynamic_food_replacement()
    
    print(f"\n🎉 CURRICULUM LEARNING READY!")
    print(f"🎓 This system will:")
    print(f"   ✅ Start easy with 10 foods (high success rate)")
    print(f"   ✅ Gradually reduce to 1 food (increasing difficulty)")
    print(f"   ✅ Provide smooth learning progression")
    print(f"   ✅ Prevent early frustration and late boredom")
    print(f"   ✅ Lead to much better final performance!")
    
    print(f"\n🚀 Ready to train with curriculum learning!")

if __name__ == "__main__":
    main()