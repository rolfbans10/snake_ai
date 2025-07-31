#!/usr/bin/env python3
"""
Test the new starvation death feature
"""

from snake_env import SnakeEnvironment
import time

def test_starvation_feature():
    print("🧪 TESTING STARVATION DEATH FEATURE")
    print("=" * 45)
    
    # Create environment and show current settings
    env = SnakeEnvironment()
    
    print(f"📊 STARVATION CONFIGURATION:")
    print(f"   💀 Death Penalty: {env.DEATH_PENALTY}")
    print(f"   🍎 Food Reward: {env.FOOD_REWARD}")
    print(f"   ⏱️ Step Penalty: {env.STEP_PENALTY}")
    print(f"   😰 Hunger Max: {env.HUNGER_MAX} (for penalty scaling)")
    print(f"   ☠️ STARVATION LIMIT: {env.MAX_STARVATION_STEPS} steps")
    print("=" * 45)
    
    print(f"🔥 STARVATION TEST:")
    print(f"Snake will be forced to die after {env.MAX_STARVATION_STEPS} steps without food!")
    print(f"Let's simulate actions and watch it happen...")
    
    obs, info = env.reset()
    steps = 0
    
    print(f"\n🐍 Starting test - snake will move randomly until starvation...")
    
    while True:
        # Take random action (just to demonstrate)
        import random
        action = random.randint(0, 3)
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        # Show progress every 10 steps
        if steps % 10 == 0:
            hunger = info['hunger']
            print(f"   Step {steps}: Hunger = {hunger}/{env.MAX_STARVATION_STEPS}")
        
        # Check if we died
        if terminated:
            print(f"\n💀 SNAKE DIED!")
            print(f"   Total steps: {steps}")
            print(f"   Hunger level: {info['hunger']}")
            print(f"   Starved to death: {info['starved_to_death']}")
            print(f"   Final score: {info['score']}")
            
            if info['starved_to_death']:
                print(f"   ✅ SUCCESS: Starvation death triggered at {info['hunger']} steps!")
            else:
                print(f"   🤔 Died from collision, not starvation")
            
            break
    
    print(f"\n🎉 STARVATION FEATURE TEST COMPLETE!")
    print(f"The snake will now be forced to find food within {env.MAX_STARVATION_STEPS} steps!")

if __name__ == "__main__":
    test_starvation_feature()