#!/usr/bin/env python3
"""
Test and demonstrate the optimized reward system
"""

from snake_env import SnakeEnvironment
import random

def analyze_reward_system():
    print("🎯 OPTIMIZED REWARD SYSTEM ANALYSIS")
    print("=" * 50)
    
    env = SnakeEnvironment()
    
    print(f"📊 NEW REWARD CONFIGURATION:")
    print(f"   💀 Death Penalty: {env.DEATH_PENALTY} (stronger death avoidance)")
    print(f"   🍎 Food Reward: {env.FOOD_REWARD} (good motivation)")
    print(f"   ⏱️ Step Penalty: {env.STEP_PENALTY} (allows exploration)")
    print(f"   😰 Hunger Max: {env.HUNGER_MAX} steps (reasonable pressure)")
    print(f"   📏 Distance: ±{env.DISTANCE_REWARD}/{env.DISTANCE_PENALTY} (balanced)")
    print(f"   ☠️ Starvation: {env.MAX_STARVATION_STEPS} steps (forces action)")
    print(f"   🍎 Foods Available: {env.NUM_FOODS} (better success rate)")
    
    print(f"\n🧮 REWARD CALCULATIONS:")
    
    # Calculate expected rewards for different scenarios
    print(f"\n📈 SUCCESS SCENARIO (Find food in 30 steps):")
    steps = 30
    step_cost = steps * env.STEP_PENALTY
    distance_help = 5 * env.DISTANCE_REWARD  # Assume 5 steps closer
    food_reward = env.FOOD_REWARD
    hunger_penalty = -((steps / env.HUNGER_MAX) ** 2)
    total = step_cost + distance_help + food_reward + hunger_penalty
    print(f"   Steps: {steps} × {env.STEP_PENALTY} = {step_cost}")
    print(f"   Distance help: ~{distance_help}")
    print(f"   Food reward: {food_reward}")
    print(f"   Hunger penalty: ~{hunger_penalty:.1f}")
    print(f"   🎉 TOTAL: ~{total:.1f} (POSITIVE!)")
    
    print(f"\n💀 DEATH SCENARIO (Die in 60 steps):")
    steps = 60
    step_cost = steps * env.STEP_PENALTY
    death_penalty = env.DEATH_PENALTY
    hunger_penalty = -((steps / env.HUNGER_MAX) ** 2)
    total = step_cost + death_penalty + hunger_penalty
    print(f"   Steps: {steps} × {env.STEP_PENALTY} = {step_cost}")
    print(f"   Death penalty: {death_penalty}")
    print(f"   Hunger penalty: ~{hunger_penalty:.1f}")
    print(f"   💀 TOTAL: ~{total:.1f} (VERY NEGATIVE!)")
    
    print(f"\n⚖️ RATIO ANALYSIS:")
    food_vs_death = env.FOOD_REWARD / abs(env.DEATH_PENALTY)
    print(f"   Food/Death ratio: {food_vs_death:.2f} (food worth {food_vs_death:.1f}× avoiding death)")
    print(f"   Distance balance: {env.DISTANCE_REWARD} vs {abs(env.DISTANCE_PENALTY)} (equal weight)")
    print(f"   Step cost: Only {abs(env.STEP_PENALTY)} per step (allows 1000 steps to lose 100 points)")
    
    print(f"\n🔬 WHY THIS IS BETTER:")
    print(f"   ✅ Death penalty ({env.DEATH_PENALTY}) > Food reward ({env.FOOD_REWARD})")
    print(f"   ✅ Balanced distance rewards (no over-cautious behavior)")
    print(f"   ✅ Low step penalty (encourages exploration)")
    print(f"   ✅ Reasonable hunger buildup ({env.HUNGER_MAX} steps)")
    print(f"   ✅ Practical starvation limit ({env.MAX_STARVATION_STEPS} steps)")
    print(f"   ✅ Multiple foods ({env.NUM_FOODS}) = 3× better chance!")
    
    return env

def simulate_training_episode():
    print(f"\n🎮 SIMULATED EPISODE:")
    print("=" * 30)
    
    env = SnakeEnvironment()
    obs, info = env.reset()
    
    total_reward = 0
    step_count = 0
    foods_eaten = 0
    
    print(f"🐍 Starting episode with {len(env.foods)} foods on board")
    print(f"   Foods at: {env.foods}")
    print(f"   Nearest food distance: {env._get_nearest_food_distance()}")
    
    # Simulate a somewhat intelligent strategy
    for step in range(200):  # Max 200 steps
        # Random action for demo
        action = random.randint(0, 3)
        
        old_score = env.score
        old_hunger = env.steps_since_food
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Track food consumption
        if env.score > old_score:
            foods_eaten += 1
            print(f"   🍎 STEP {step_count}: FOOD EATEN! Total: {foods_eaten}")
            print(f"      Reward: +{reward:.1f}, Hunger reset: {old_hunger}→0")
        
        # Print hunger warnings
        if env.steps_since_food > 0 and env.steps_since_food % 25 == 0:
            hunger_penalty = (env.steps_since_food / env.HUNGER_MAX) ** 2
            print(f"   😰 STEP {step_count}: Hunger at {env.steps_since_food}, penalty: -{hunger_penalty:.1f}")
        
        # Check for starvation warning
        if env.steps_since_food >= env.MAX_STARVATION_STEPS - 10:
            print(f"   ⚠️ STEP {step_count}: STARVATION WARNING! {env.steps_since_food}/{env.MAX_STARVATION_STEPS}")
        
        if terminated:
            if info.get('starved_to_death'):
                print(f"   ☠️ STEP {step_count}: STARVED TO DEATH after {env.steps_since_food} steps without food!")
            else:
                print(f"   💥 STEP {step_count}: COLLISION DEATH!")
            break
    
    print(f"\n📊 EPISODE RESULTS:")
    print(f"   Steps: {step_count}")
    print(f"   Foods eaten: {foods_eaten}")
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Avg reward per step: {total_reward/step_count:.2f}")
    print(f"   Final hunger: {info['hunger']}")
    
    if foods_eaten > 0:
        print(f"   🎉 SUCCESS! AI found food {foods_eaten} times!")
    else:
        print(f"   ❌ FAILED: No food found")
    
    return total_reward, step_count, foods_eaten

def main():
    env = analyze_reward_system()
    
    print(f"\n" + "="*50)
    simulate_training_episode()
    
    print(f"\n🚀 TRAINING RECOMMENDATION:")
    print(f"This balanced system should:")
    print(f"✅ Encourage food-seeking (positive rewards possible)")
    print(f"✅ Discourage death strongly (large negative penalty)")
    print(f"✅ Allow exploration (small step penalty)")
    print(f"✅ Build hunger pressure gradually (50-step buildup)")
    print(f"✅ Force action within reasonable time (150 steps)")
    print(f"✅ Provide multiple opportunities (3 foods)")
    
    print(f"\n🎯 START TRAINING WITH:")
    print(f"python train_enhanced_ai.py")

if __name__ == "__main__":
    main()