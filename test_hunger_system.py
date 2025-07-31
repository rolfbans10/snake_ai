#!/usr/bin/env python3
"""
Test the new hunger/starvation system
"""

from snake_env import SnakeEnvironment
import time

def test_hunger_system():
    print("🍽️  TESTING HUNGER/STARVATION SYSTEM")
    print("=" * 50)
    
    env = SnakeEnvironment()
    
    print("🧪 How the hunger system works:")
    print("   - Hunger penalty starts at 0")
    print("   - Grows progressively: (steps/100)²")
    print("   - Resets to 0 when food is eaten")
    print("   - Creates urgency to find food!")
    
    # Test 1: Show hunger progression without food
    print(f"\n📊 HUNGER PROGRESSION (without food):")
    for steps in [0, 10, 25, 50, 75, 90, 100]:
        penalty = (steps / 100) ** 2
        print(f"   Steps {steps:3d}: Hunger penalty = -{penalty:.3f}")
    
    print(f"\n📈 Key Insight:")
    print(f"   - Early steps: Tiny penalty (can explore safely)")
    print(f"   - Later steps: BIG penalty (must find food!)")
    print(f"   - After 100 steps: -1.0 penalty per step!")
    
    # Test 2: Simulate AI behavior
    print(f"\n🎮 SIMULATING AI BEHAVIOR:")
    obs, info = env.reset()
    total_reward = 0
    
    print(f"Initial state: Score={info['score']}, Hunger={info['hunger']}")
    
    for step in range(15):
        # Simulate random movement (not finding food)
        action = step % 4  # cycle through actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"Step {step+1:2d}: {action_names[action]} → Reward: {reward:+6.3f}, Hunger: {info['hunger']}, Total: {total_reward:+6.3f}")
        
        if terminated:
            print("💀 Snake died!")
            break
    
    print(f"\n🔍 ANALYSIS:")
    print(f"   Notice how reward gets MORE NEGATIVE over time?")
    print(f"   This forces AI to take RISKS to find food!")
    print(f"   No more safe circling - AI must be AGGRESSIVE!")

def demo_food_finding():
    print(f"\n\n🍎 DEMO: What happens when food IS found:")
    print("=" * 50)
    
    env = SnakeEnvironment()
    obs, info = env.reset()
    
    # Simulate 30 steps without food, then finding food
    print("Simulating hungry snake that finally finds food...")
    
    # Build up hunger
    for step in range(30):
        obs, reward, terminated, truncated, info = env.step(0)  # Keep moving up
        if step % 10 == 9:  # Show progress every 10 steps
            print(f"Step {step+1}: Hunger={info['hunger']}, Penalty={info['hunger_penalty']:.3f}")
        if terminated:
            print("Snake died before finding food!")
            return
    
    print(f"\n🎯 MOMENT OF TRUTH - Simulating food discovery:")
    
    # Manually simulate finding food
    env.score += 1  # Simulate eating food
    obs, reward, terminated, truncated, info = env.step(0)
    
    print(f"   BEFORE eating: High hunger penalty")
    print(f"   AFTER eating:  Reward = +{reward:.1f}, Hunger = {info['hunger']} (RESET!)")
    print(f"   🎉 Success! AI gets HUGE reward and hunger relief!")

if __name__ == "__main__":
    test_hunger_system()
    demo_food_finding()
    
    print(f"\n🚀 READY TO TRAIN:")
    print(f"   The AI can no longer survive by circling!")
    print(f"   It MUST find food or face growing penalties!")
    print(f"   Run: python train_hungry_ai.py")