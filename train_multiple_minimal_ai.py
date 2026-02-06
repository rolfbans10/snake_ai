#!/usr/bin/env python3
"""
Multi-Experiment Training Orchestrator for Snake AI
Runs multiple training experiments in parallel with different configurations.

Usage:
    python train_multiple_minimal_ai.py                    # Run default experiments
    python train_multiple_minimal_ai.py --parallel 2      # Run 2 at a time
    python train_multiple_minimal_ai.py --device cuda     # Use GPU for all
    python train_multiple_minimal_ai.py --dry-run         # Show what would run
"""

import subprocess
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Experiment:
    """Configuration for a single training experiment"""
    name: str
    arch: List[int]
    steps: int
    device: str = "cpu"
    
    def get_command(self) -> List[str]:
        """Generate the command to run this experiment"""
        arch_args = " ".join(map(str, self.arch))
        return [
            "python", "train_minimal_ai.py",
            "--arch", *map(str, self.arch),
            "--steps", str(self.steps),
            "--device", self.device,
            "--name", self.name
        ]
    
    def __str__(self) -> str:
        arch_str = "x".join(map(str, self.arch))
        return f"{self.name} [{arch_str}] ({self.steps:,} steps, {self.device})"


# ============================================================================
# DEFINE YOUR EXPERIMENTS HERE
# ============================================================================

def get_default_experiments(device: str = "cpu", steps: int = 250000) -> List[Experiment]:
    """
    Define the experiments to run.
    Modify this function to customize your experiments!
    """
    return [
        # Tiny networks - fast training, might underfit
        Experiment(name="tiny_10x10", arch=[10, 10], steps=steps, device=device),
        Experiment(name="tiny_10x10x10", arch=[10, 10, 10], steps=steps, device=device),
        
        # Small networks - good balance
        Experiment(name="small_28x28", arch=[28, 28], steps=steps, device=device),
        Experiment(name="small_28x28x28", arch=[28, 28, 28], steps=steps, device=device),
        
        # Medium networks - standard choice
        Experiment(name="medium_64x64", arch=[64, 64], steps=steps, device=device),
        Experiment(name="medium_64x64x64", arch=[64, 64, 64], steps=steps, device=device),
        
        # Larger networks - more capacity
        Experiment(name="large_128x128", arch=[128, 128], steps=steps, device=device),
        
        # Deep networks - test depth vs width
        Experiment(name="deep_32x32x32x32", arch=[32, 32, 32, 32], steps=steps, device=device),
    ]


def get_quick_experiments(device: str = "cpu") -> List[Experiment]:
    """Quick experiments for testing (fewer steps)"""
    return [
        Experiment(name="quick_10x10", arch=[10, 10], steps=50000, device=device),
        Experiment(name="quick_28x28", arch=[28, 28], steps=50000, device=device),
        Experiment(name="quick_64x64", arch=[64, 64], steps=50000, device=device),
    ]


def get_depth_experiments(device: str = "cpu", steps: int = 250000) -> List[Experiment]:
    """Experiments focused on network depth"""
    return [
        Experiment(name="depth_64x1", arch=[64], steps=steps, device=device),
        Experiment(name="depth_64x2", arch=[64, 64], steps=steps, device=device),
        Experiment(name="depth_64x3", arch=[64, 64, 64], steps=steps, device=device),
        Experiment(name="depth_64x4", arch=[64, 64, 64, 64], steps=steps, device=device),
        Experiment(name="depth_64x5", arch=[64, 64, 64, 64, 64], steps=steps, device=device),
    ]


def get_width_experiments(device: str = "cpu", steps: int = 250000) -> List[Experiment]:
    """Experiments focused on network width"""
    return [
        Experiment(name="width_16x16", arch=[16, 16], steps=steps, device=device),
        Experiment(name="width_32x32", arch=[32, 32], steps=steps, device=device),
        Experiment(name="width_64x64", arch=[64, 64], steps=steps, device=device),
        Experiment(name="width_128x128", arch=[128, 128], steps=steps, device=device),
        Experiment(name="width_256x256", arch=[256, 256], steps=steps, device=device),
    ]


# ============================================================================
# ORCHESTRATION LOGIC
# ============================================================================

def run_experiment(experiment: Experiment, live_output: bool = False, log_dir: str = "logs/experiments") -> dict:
    """Run a single experiment and return results"""
    start_time = time.time()
    cmd = experiment.get_command()
    
    print(f"🚀 Starting: {experiment}")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{experiment.name}.log"
    
    try:
        if live_output:
            # Stream output directly to console (for sequential mode)
            print(f"   📝 Live output enabled for {experiment.name}")
            print("-" * 50)
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                text=True
            )
            print("-" * 50)
            stdout, stderr = "", ""
        else:
            # Capture output and write to log file (for parallel mode)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)) or "."
            )
            stdout, stderr = result.stdout, result.stderr
            
            # Write output to log file
            with open(log_file, "w") as f:
                f.write(f"=== Experiment: {experiment.name} ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write("=== STDOUT ===\n")
                f.write(stdout or "(empty)\n")
                f.write("\n=== STDERR ===\n")
                f.write(stderr or "(empty)\n")
            
            print(f"   📝 Log: {log_file}")
        
        elapsed = time.time() - start_time
        success = result.returncode == 0
        
        if success:
            print(f"✅ Completed: {experiment.name} ({elapsed/60:.1f} min)")
        else:
            print(f"❌ Failed: {experiment.name}")
            if stderr:
                print(f"   Error: {stderr[:500]}")
        
        return {
            "name": experiment.name,
            "success": success,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "log_file": log_file
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Exception in {experiment.name}: {e}")
        return {
            "name": experiment.name,
            "success": False,
            "elapsed_seconds": elapsed,
            "error": str(e)
        }


def run_experiments_parallel(experiments: List[Experiment], max_parallel: int = 2) -> List[dict]:
    """Run multiple experiments in parallel"""
    results = []
    
    print(f"\n🔬 Running {len(experiments)} experiments ({max_parallel} in parallel)")
    print("=" * 60)
    print(f"📝 Logs will be written to: logs/experiments/")
    print(f"👀 Monitor progress with: tail -f logs/experiments/<name>.log")
    print("")
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        future_to_exp = {executor.submit(run_experiment, exp, False): exp for exp in experiments}
        
        for future in as_completed(future_to_exp):
            result = future.result()
            results.append(result)
    
    return results


def run_experiments_sequential(experiments: List[Experiment], live_output: bool = False) -> List[dict]:
    """Run experiments one at a time"""
    results = []
    
    mode = "LIVE OUTPUT" if live_output else "sequential"
    print(f"\n🔬 Running {len(experiments)} experiments ({mode})")
    print("=" * 60)
    
    if not live_output:
        print(f"📝 Logs will be written to: logs/experiments/")
        print(f"💡 Use --live for real-time output")
        print("")
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        result = run_experiment(experiment, live_output=live_output)
        results.append(result)
    
    return results


def print_summary(results: List[dict]):
    """Print summary of all experiments"""
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        mins = r["elapsed_seconds"] / 60
        print(f"   - {r['name']}: {mins:.1f} minutes")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   - {r['name']}: {r.get('error', 'See logs')}")
    
    total_time = sum(r["elapsed_seconds"] for r in results)
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    
    print("\n📁 Models saved in: models/")
    print("📊 Logs saved in: logs/")
    print("📈 View TensorBoard: tensorboard --logdir logs/tensorboard/")
    
    return [r["name"] for r in successful]


def compare_trained_models(model_names: List[str], episodes: int = 20):
    """Compare all trained models and rank them"""
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON")
    print("=" * 60)
    print(f"\nTesting each model for {episodes} episodes...")
    
    # Import here to avoid issues if run before training
    from snake_env_simple_minimal import MinimalSnakeEnvironment
    from stable_baselines3 import PPO
    
    results = []
    
    for model_name in model_names:
        model_path = f"models/{model_name}"
        
        if not os.path.exists(f"{model_path}.zip"):
            print(f"   ⚠️  Skipping {model_name} - model file not found")
            continue
        
        print(f"\n   Testing: {model_name}...")
        
        try:
            model = PPO.load(model_path)
            env = MinimalSnakeEnvironment()
            
            scores = []
            steps_list = []
            
            for episode in range(episodes):
                obs, info = env.reset()
                episode_steps = 0
                
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    
                    if terminated or truncated or episode_steps > 2000:
                        scores.append(info['score'])
                        steps_list.append(episode_steps)
                        break
            
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            avg_steps = sum(steps_list) / len(steps_list)
            
            results.append({
                "name": model_name,
                "avg_score": avg_score,
                "max_score": max_score,
                "avg_steps": avg_steps,
                "scores": scores
            })
            
            print(f"      Avg Score: {avg_score:.2f}, Max: {max_score}, Avg Steps: {avg_steps:.1f}")
            
        except Exception as e:
            print(f"   ❌ Error testing {model_name}: {e}")
    
    if not results:
        print("\n⚠️  No models could be tested!")
        return
    
    # Sort by average score (descending)
    results.sort(key=lambda x: x["avg_score"], reverse=True)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 FINAL RANKINGS")
    print("=" * 70)
    print(f"{'Rank':<6} {'Model':<30} {'Avg Score':>10} {'Max Score':>10} {'Avg Steps':>10}")
    print("-" * 70)
    
    medals = ["🥇", "🥈", "🥉"]
    
    for i, r in enumerate(results):
        rank_str = medals[i] if i < 3 else f"  {i+1}."
        print(f"{rank_str:<6} {r['name']:<30} {r['avg_score']:>10.2f} {r['max_score']:>10} {r['avg_steps']:>10.1f}")
    
    # Declare winner
    winner = results[0]
    print("\n" + "=" * 70)
    print(f"🏆 WINNER: {winner['name']}")
    print(f"   Average Score: {winner['avg_score']:.2f}")
    print(f"   Best Score: {winner['max_score']}")
    print(f"   Average Survival: {winner['avg_steps']:.1f} steps")
    print("=" * 70)
    
    # Performance insights
    print("\n💡 INSIGHTS:")
    
    if len(results) >= 2:
        score_diff = results[0]['avg_score'] - results[-1]['avg_score']
        print(f"   Score range: {results[-1]['avg_score']:.2f} to {results[0]['avg_score']:.2f} (diff: {score_diff:.2f})")
    
    if winner['avg_score'] >= 5:
        print(f"   🎯 Winner performs EXCELLENTLY!")
    elif winner['avg_score'] >= 2:
        print(f"   👍 Winner performs WELL!")
    elif winner['avg_score'] >= 1:
        print(f"   📈 Winner is LEARNING - consider more training")
    else:
        print(f"   🔄 All models need more training")
    
    print(f"\n📋 Watch the winner play:")
    print(f"   python watch_minimal_ai.py --model {winner['name']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="🐍 Multi-Experiment Snake AI Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Sets:
  default    Standard architecture comparison (8 experiments)
  quick      Fast test run with fewer steps (3 experiments)
  depth      Compare network depths with fixed width (5 experiments)
  width      Compare network widths with fixed depth (5 experiments)

Examples:
  python train_multiple_minimal_ai.py                           # Run default experiments
  python train_multiple_minimal_ai.py --parallel 4              # Run 4 at a time
  python train_multiple_minimal_ai.py --experiments quick       # Quick test
  python train_multiple_minimal_ai.py --device cuda --parallel 2
  python train_multiple_minimal_ai.py --dry-run                 # Preview only
  python train_multiple_minimal_ai.py --steps 500000            # More training
  python train_multiple_minimal_ai.py --live                    # See live training output
  python train_multiple_minimal_ai.py --test-episodes 50        # More thorough comparison
  python train_multiple_minimal_ai.py --no-compare              # Skip final comparison

Monitoring parallel experiments:
  tail -f logs/experiments/*.log                                # Watch all logs
  tail -f logs/experiments/tiny_10x10.log                       # Watch specific experiment
        """
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        choices=["default", "quick", "depth", "width"],
        default="default",
        help="Which experiment set to run (default: default)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Number of experiments to run in parallel (default: 2)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for all experiments (default: cpu)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=250000,
        help="Training steps for each experiment (default: 250000)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments one at a time instead of parallel"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show live output (implies --sequential)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running"
    )
    
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip model comparison at the end"
    )
    
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=20,
        help="Number of episodes for final comparison (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Select experiment set
    if args.experiments == "quick":
        experiments = get_quick_experiments(args.device)
    elif args.experiments == "depth":
        experiments = get_depth_experiments(args.device, args.steps)
    elif args.experiments == "width":
        experiments = get_width_experiments(args.device, args.steps)
    else:
        experiments = get_default_experiments(args.device, args.steps)
    
    # Print experiment plan
    print("🐍 MULTI-EXPERIMENT SNAKE AI TRAINING")
    print("=" * 60)
    print(f"📋 Experiment set: {args.experiments}")
    print(f"🔢 Total experiments: {len(experiments)}")
    print(f"💻 Device: {args.device}")
    if args.live:
        print(f"🔀 Mode: LIVE OUTPUT (sequential with real-time display)")
    elif args.sequential:
        print(f"🔀 Mode: Sequential (one at a time)")
    else:
        print(f"🔀 Mode: Parallel ({args.parallel} at a time)")
    print(f"📊 Steps per experiment: {args.steps:,}")
    
    print("\n📝 Experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"   {i}. {exp}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Commands that would be executed:")
        for exp in experiments:
            print(f"   {' '.join(exp.get_command())}")
        print("\n✋ Dry run complete. Remove --dry-run to execute.")
        return
    
    # Confirm before running
    print(f"\n⏳ Estimated time: ~{len(experiments) * 10 / args.parallel:.0f}+ minutes")
    
    # Run experiments
    start_time = time.time()
    
    if args.live:
        # Live output implies sequential
        results = run_experiments_sequential(experiments, live_output=True)
    elif args.sequential:
        results = run_experiments_sequential(experiments, live_output=False)
    else:
        results = run_experiments_parallel(experiments, args.parallel)
    
    # Print summary
    successful_models = print_summary(results)
    
    # Compare all trained models
    if successful_models and not args.dry_run and not args.no_compare:
        compare_trained_models(successful_models, episodes=args.test_episodes)
    elif args.no_compare:
        print("\n⏭️  Skipping model comparison (--no-compare)")
        print("   Run manually: python watch_minimal_ai.py --compare " + " ".join(successful_models))
    
    print("\n🎉 All experiments complete!")


if __name__ == "__main__":
    main()
