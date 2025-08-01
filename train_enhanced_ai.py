#!/usr/bin/env python3
"""
Train AI with Enhanced Reward System
- Distance rewards for directional guidance  
- Configurable reward parameters
- Multiple pressure systems
"""

from snake_env import SnakeEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

TRAINING_STEPS = 1_000_000

def main():
    print("🚀 TRAINING WITH ENHANCED REWARD SYSTEM!")
    print("=" * 50)
    
    # Create environment and show configuration
    base_env = SnakeEnvironment()
    env = Monitor(base_env, "logs_enhanced/")
    
    print(f"📊 REWARD CONFIGURATION:")
    print(f"   💀 Death Penalty: {base_env.DEATH_PENALTY}")
    print(f"   🍎 Food Reward: {base_env.FOOD_REWARD}")
    print(f"   ⏱️ Step Penalty: {base_env.STEP_PENALTY}")
    print(f"   📏 Distance: +{base_env.DISTANCE_REWARD} closer / {base_env.DISTANCE_PENALTY} farther")
    print(f"   😰 Hunger Max: {base_env.HUNGER_MAX} steps")
    print("=" * 50)
    
    # Create fresh AI model
    print(f"🧠 Creating fresh AI model...")
    
    # Check for GPU availability
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Optimize GPU settings for maximum utilization
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for faster computation
        torch.backends.cudnn.allow_tf32 = True  # Use TensorFloat-32 for faster computation
        
        print("⚡ GPU optimizations enabled:")
        print("   - cuDNN benchmark mode")
        print("   - TensorFloat-32 acceleration")
        print("   - Non-deterministic algorithms for speed")
        print("   - Maximum memory allocation")
        
        # Set additional optimizations
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Force GPU to maximum performance
        torch.cuda.set_device(0)
        torch.cuda.synchronize()  # Ensure GPU is ready
        
        # Additional GPU optimizations
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Set compute mode for maximum GPU utilization
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print(f"💻 Using CPU (GPU not available)")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=8192,  # Reduced to reduce CPU bottleneck
        batch_size=4096,  # Much larger batch size for GPU
        device=device,
        n_epochs=10,  # Single epoch to reduce CPU overhead
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        target_kl=None,  # Early stopping if KL divergence is too high
        tensorboard_log="logs_enhanced/",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 1024, 1024],  # Massive policy network for GPU
                vf=[1024, 1024, 1024]   # Massive value network for GPU
            ),
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
    )
    
    print(f"✅ Model created!")
    
    # Train with enhanced rewards
    print(f"\n🏋️ TRAINING SESSION:")
    print(f"Training for {TRAINING_STEPS:,} steps with enhanced rewards...")
    
    # GPU monitoring and optimization
    if torch.cuda.is_available():
        print(f"📊 GPU Memory before training: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"📊 GPU Memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set memory fraction to use more GPU memory
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("⚡ Flash Attention enabled for better GPU utilization")
        
        # Enable additional GPU optimizations
        torch.backends.cuda.enable_math_sdp(True)  # Enable math SDP for better performance
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Enable memory efficient SDP
        
        # Set higher memory allocation
        torch.cuda.empty_cache()  # Clear cache before starting
        torch.cuda.memory.empty_cache()  # Additional cache clearing
        
        # Force GPU computation and reduce CPU bottleneck
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set environment variables to reduce CPU usage
        import os
        os.environ['OMP_NUM_THREADS'] = '1'  # Reduce OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '1'  # Reduce MKL threads
        os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Reduce NumExpr threads
        
        print("🚀 GPU-focused optimizations enabled:")
        print("   - 95% GPU memory usage")
        print("   - Math SDP enabled")
        print("   - Memory efficient SDP enabled")
        print("   - CPU threads limited to reduce bottleneck")
        print("   - TF32 acceleration enabled")
        print("   - cuDNN benchmark mode")
    
    model.learn(
        total_timesteps=TRAINING_STEPS,
        progress_bar=True
    )
    
    if torch.cuda.is_available():
        print(f"📊 GPU Memory after training: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        torch.cuda.empty_cache()  # Clear GPU memory
    
    print(f"✅ Training completed!")
    
    # Save the model
    print(f"\n💾 Saving enhanced AI...")
    model.save("models/enhanced_reward_ai")
    print(f"✅ Saved as 'enhanced_reward_ai'")
    
    # Quick test
    print(f"\n🧪 QUICK PERFORMANCE TEST:")
    test_episodes = 3
    total_scores = []
    total_steps = []
    
    for episode in range(test_episodes):
        obs, info = env.reset()
        episode_steps = 0
        episode_score = 0
        
        while episode_steps < 150:  # Limit episode length
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated or truncated:
                episode_score = info['score']
                break
        
        total_scores.append(episode_score)
        total_steps.append(episode_steps)
        
        print(f"   Episode {episode+1}: Score={episode_score}, Steps={episode_steps}")
    
    avg_score = np.mean(total_scores)
    avg_steps = np.mean(total_steps)
    
    print(f"\n📈 RESULTS:")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Total Foods Found: {sum(total_scores)}")
    
    if avg_score > 0.5:
        print(f"   🏆 EXCELLENT: AI is finding food consistently!")
    elif avg_score > 0:
        print(f"   🎯 GOOD: AI found some food!")
    elif avg_steps > 50:
        print(f"   📈 PROMISING: AI is exploring (not dying quickly)!")
    else:
        print(f"   🔧 LEARNING: AI needs more training")
    
    print(f"\n🎉 ENHANCED TRAINING COMPLETE!")
    print(f"The AI now has:")
    print(f"   - Distance guidance ✅")
    print(f"   - Hunger pressure ✅") 
    print(f"   - Death avoidance ✅")
    print(f"   - Efficiency pressure ✅")
    print(f"   - Big food rewards ✅")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_enhanced", exist_ok=True)
    
    main()