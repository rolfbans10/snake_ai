#!/usr/bin/env python3
"""
Test the tweaked reward system based on training analysis
"""

from snake_env import SnakeEnvironment

def analyze_tweaks():
    print("🔧 TWEAKED REWARD SYSTEM ANALYSIS")
    print("=" * 50)
    
    env = SnakeEnvironment()
    
    print(f"📈 CHANGES MADE BASED ON TRAINING RESULTS:")
    print(f"   ⏱️ Step Penalty: -0.1 → -0.05 (50% less harsh)")
    print(f"   😰 Hunger Max: 50 → 75 steps (50% more time)")  
    print(f"   📏 Distance: ±1.0 → ±2.0 (2× stronger guidance)")
    print(f"   ☠️ Starvation: 150 → 250 steps (67% more time)")
    print(f"   🍎 Foods: 3 (unchanged - working well)")
    
    print(f"\n📊 NEW CONFIGURATION:")
    print(f"   💀 Death Penalty: {env.DEATH_PENALTY}")
    print(f"   🍎 Food Reward: {env.FOOD_REWARD}")
    print(f"   ⏱️ Step Penalty: {env.STEP_PENALTY}")
    print(f"   😰 Hunger Max: {env.HUNGER_MAX} steps")
    print(f"   📏 Distance: ±{env.DISTANCE_REWARD}/{env.DISTANCE_PENALTY}")
    print(f"   ☠️ Starvation: {env.MAX_STARVATION_STEPS} steps")
    
    print(f"\n🧮 IMPACT ANALYSIS:")
    
    # Scenario 1: Long exploration
    print(f"\n📈 LONG EXPLORATION SCENARIO (200 steps to find food):")
    steps = 200
    step_cost = steps * env.STEP_PENALTY
    food_reward = env.FOOD_REWARD
    hunger_penalty = -((steps / env.HUNGER_MAX) ** 2)
    distance_help = 10 * env.DISTANCE_REWARD  # Assume 10 steps closer
    total = step_cost + food_reward + hunger_penalty + distance_help
    
    print(f"   Steps: {steps} × {env.STEP_PENALTY} = {step_cost}")
    print(f"   Food reward: {food_reward}")
    print(f"   Hunger penalty: ~{hunger_penalty:.1f}")
    print(f"   Distance help: ~{distance_help}")
    print(f"   🎉 TOTAL: ~{total:.1f} (STILL POSITIVE!)")
    
    # Scenario 2: Medium survival
    print(f"\n⚖️ MEDIUM SURVIVAL (100 steps, no food):")
    steps = 100
    step_cost = steps * env.STEP_PENALTY
    hunger_penalty = -((steps / env.HUNGER_MAX) ** 2)
    distance_penalty = 5 * env.DISTANCE_PENALTY  # Assume some wrong moves
    total = step_cost + hunger_penalty + distance_penalty
    
    print(f"   Steps: {steps} × {env.STEP_PENALTY} = {step_cost}")
    print(f"   Hunger penalty: ~{hunger_penalty:.1f}")
    print(f"   Distance penalty: ~{distance_penalty}")
    print(f"   📊 TOTAL: ~{total:.1f} (Mild negative - encourages better strategy)")
    
    # Scenario 3: Starvation death
    print(f"\n☠️ STARVATION DEATH SCENARIO ({env.MAX_STARVATION_STEPS} steps):")
    steps = env.MAX_STARVATION_STEPS
    step_cost = steps * env.STEP_PENALTY
    death_penalty = env.DEATH_PENALTY
    hunger_penalty = -((steps / env.HUNGER_MAX) ** 2)
    total = step_cost + death_penalty + hunger_penalty
    
    print(f"   Steps: {steps} × {env.STEP_PENALTY} = {step_cost}")
    print(f"   Death penalty: {death_penalty}")
    print(f"   Hunger penalty: ~{hunger_penalty:.1f}")
    print(f"   💀 TOTAL: ~{total:.1f} (VERY NEGATIVE!)")
    
    print(f"\n✅ WHY THESE TWEAKS SHOULD WORK:")
    print(f"   🔹 Smaller step penalty allows 250+ step explorations")
    print(f"   🔹 Slower hunger buildup gives AI time to learn navigation")
    print(f"   🔹 Stronger distance rewards provide clearer guidance")
    print(f"   🔹 Generous starvation limit prevents premature deaths")
    print(f"   🔹 Multiple foods still provide opportunities")
    
    print(f"\n🎯 EXPECTED IMPROVEMENTS:")
    print(f"   📈 Longer episodes (250+ steps possible)")
    print(f"   🎯 Better food-finding (stronger distance guidance)")
    print(f"   🧠 More learning time (generous time limits)")
    print(f"   ⚡ Better test performance (less aggressive penalties)")
    
    return env

def quick_simulation():
    print(f"\n🎮 QUICK SIMULATION:")
    print("=" * 25)
    
    env = SnakeEnvironment()
    obs, info = env.reset()
    
    print(f"🐍 Starting with {len(env.foods)} foods")
    print(f"   Foods: {env.foods}")
    print(f"   Snake: {env.snake[0]}")
    print(f"   Nearest distance: {env._get_nearest_food_distance()}")
    
    total_reward = 0
    
    # Simulate 50 steps
    import random
    for step in range(50):
        action = random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"   Step {step}: Reward={reward:.2f}, Hunger={info['hunger']}, Distance={env._get_nearest_food_distance()}")
        
        if terminated:
            break
    
    print(f"\n📊 50-STEP SIMULATION:")
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Average per step: {total_reward/min(step+1, 50):.2f}")
    print(f"   Final hunger: {info['hunger']}")
    print(f"   Foods eaten: {env.score}")

def main():
    env = analyze_tweaks()
    quick_simulation()
    
    print(f"\n🚀 READY FOR IMPROVED TRAINING!")
    print(f"These tweaks address the key issues from your training:")
    print(f"✅ Less aggressive penalties = more exploration")
    print(f"✅ Stronger guidance = better food-finding")
    print(f"✅ More time = better learning opportunities")
    print(f"✅ Should see much better test performance!")

if __name__ == "__main__":
    main()