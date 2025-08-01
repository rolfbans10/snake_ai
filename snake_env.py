from snake_game import SnakeGame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class SnakeEnvironment(SnakeGame, gym.Env):
    """
    AI-Ready Snake Environment that extends SnakeGame and implements OpenAI Gym interface
    """
    
    # 🎯 CURRICULUM LEARNING REWARD SYSTEM (Progressive Difficulty)
    DEATH_PENALTY = -200        # Strong penalty for dying (avoid death at all costs)
    FOOD_REWARD = 100          # Good reward for eating food (main objective)  
    STEP_PENALTY = -0.05       # Even smaller step penalty (more exploration time)
    HUNGER_MAX = 75           # Slower hunger buildup (more time to learn)
    DISTANCE_REWARD = 2.0     # Stronger guidance toward food
    DISTANCE_PENALTY = -2.0   # Stronger penalty for moving away
    MAX_STARVATION_STEPS = 2500  # More generous starvation limit (learning time)
    
    # 🔍 EXPLORATION REWARDS
    EXPLORATION_REWARD = 1.0    # Reward for visiting new areas
    EXPLORATION_DECAY = 0.95    # How quickly exploration reward decays for revisited areas
    
    # 🏆 WIN CONDITION
    WIN_REWARD = 1000          # Massive reward for eating all foods (WIN CONDITION!)
    
    # 🎓 CURRICULUM LEARNING: Dynamic Food Count
    INITIAL_FOODS = 1          # Start with single food (easy wins!)
    FINAL_FOODS = 10          # End with many foods (hard challenge!)
    TOTAL_TRAINING_STEPS = 1000000   # Total training timesteps (match actual training)
    
    def __init__(self):
        # Call the parent class constructor
        super().__init__()
        
        # Define action and observation spaces for AI training
        self.action_space = spaces.Discrete(4)  # 4 actions: up, right, down, left
        
        # Observation space: flattened 24x32 grid = 768 values, each between -1 and 1
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(768,),  # 24 * 32 = 768
            dtype=np.float32
        )
        
        # 🎓 CURRICULUM LEARNING: Track training progress
        self.global_step_count = 0
        self.foods = []  # Initialize empty foods list
        self._last_food_count = self.INITIAL_FOODS  # Track food count changes
        
        # 🔍 EXPLORATION TRACKING: Track visited positions
        self.visited_positions = {}  # Track how many times each position was visited
        
        print(f"🤖 AI Environment created!")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space}")
        print(f"🎓 Curriculum Learning: {self.INITIAL_FOODS} → {self.FINAL_FOODS} foods over {self.TOTAL_TRAINING_STEPS:,} steps")
        print(f"🔍 Exploration Rewards: +{self.EXPLORATION_REWARD} for new areas, {self.EXPLORATION_DECAY}x decay for revisits")
    
    def reset(self, seed=None):
        """Reset environment for AI training"""
        super().reset()  # Use parent's reset method
        self.prev_score = 0
        
        # HUNGER SYSTEM: Track steps since last food
        self.steps_since_food = 0
        
        # DISTANCE TRACKING: Track distance to food for directional rewards
        self.prev_distance_to_food = None
        
        # 🔍 EXPLORATION TRACKING: Reset exploration for new episode
        if hasattr(self, 'visited_positions'):
            self.visited_positions.clear()  # Clear visited positions for new episode
        else:
            self.visited_positions = {}  # Initialize if not exists
        
        # MULTI-FOOD SYSTEM: Override single food with multiple foods
        self._place_multiple_foods()
        
        # Return observation and info (gym format)
        observation = self.get_state_array()
        info = {'score': self.score, 'hunger': self.steps_since_food}
        return observation, info
    
    def step(self, action):
        """
        Take one step in the environment
        action: 0=up, 1=right, 2=down, 3=left
        """
        # 🎓 CURRICULUM LEARNING: Update step counter
        self._update_step_counter()
        
        # Use parent's step method
        super().step(action)
        
        # HUNGER SYSTEM: Calculate reward with starvation pressure
        reward = 0
        
        # Increment hunger counter
        self.steps_since_food += 1
        
        # STEP 2: Food reward - check if score increased (ate food)
        ate_food = False
        if hasattr(self, 'prev_score'):
            if self.score > self.prev_score:
                reward += self.FOOD_REWARD
                ate_food = True
                # RESET HUNGER when food is eaten!
                self.steps_since_food = 0
                print(f"    🍎 FOOD FOUND! Score: {self.score}, Hunger reset. +{self.FOOD_REWARD} reward")
        
        # Store current score for next time
        self.prev_score = self.score
        
        # STEP 3: DISTANCE REWARD - encourage moving toward food
        distance_reward = self._calculate_distance_reward()
        reward += distance_reward
        
        # STEP 4: EXPLORATION REWARD - encourage visiting new areas
        exploration_reward = self._calculate_exploration_reward()
        reward += exploration_reward
        
        # STEP 5: STEP PENALTY - small penalty for each step to encourage efficiency
        reward += self.STEP_PENALTY
        
        # STEP 6: HUNGER PENALTY - grows over time without food
        if not ate_food:
            # Progressive hunger penalty that gets worse over time
            hunger_penalty = (self.steps_since_food / self.HUNGER_MAX) ** 2
            reward -= hunger_penalty
            
            # Show hunger status occasionally
            if self.steps_since_food % 25 == 0:
                print(f"    😰 HUNGER: {self.steps_since_food} steps, Score: {self.score}, penalty: -{hunger_penalty:.2f}")
        
        # STEP 7: STARVATION DEATH - force death if too many steps without food
        if self.steps_since_food >= self.MAX_STARVATION_STEPS:
            self.game_over = True  # Force death due to starvation
            print(f"    💀 STARVED TO DEATH! Score: {self.score}, {self.steps_since_food} steps without food!")
        
        # STEP 8: WIN CONDITION - check if all foods eaten
        if len(self.foods) == 0 and not self.game_over:
            self.game_over = True  # End episode on victory
            reward += self.WIN_REWARD  # Massive win bonus!
            print(f"    🏆 VICTORY! All {self.score} foods eaten! +{self.WIN_REWARD} reward!")
        
        # STEP 9: Death penalty - only if we died (not if we won)
        if self.game_over and len(self.foods) > 0:  # Only apply death penalty if we didn't win
            reward = self.DEATH_PENALTY  # Override any other rewards if we died
            
            # Determine cause of death
            if self.steps_since_food >= self.MAX_STARVATION_STEPS:
                death_cause = "STARVATION"
            else:
                # Check collision type
                head = self.snake[0]
                if (head[0] < 0 or head[0] >= self.WINDOW_WIDTH or
                    head[1] < 0 or head[1] >= self.WINDOW_HEIGHT):
                    death_cause = "WALL COLLISION"
                elif head in self.snake[1:]:
                    death_cause = "SELF COLLISION"
                else:
                    death_cause = "UNKNOWN"
            
            print(f"    💀 DEATH ({death_cause}): Score: {self.score}, Areas: {len(self.visited_positions)}, Penalty: {self.DEATH_PENALTY:.1f}")
        
        # Log total reward for this step (for debugging) - only major events
        if self.game_over and len(self.foods) == 0:  # Victory!
            print(f"    💰 VICTORY REWARD: {reward:.2f}")
        elif self.game_over:  # Death
            print(f"    💰 DEATH REWARD: {reward:.2f}")
        elif reward > 50:  # Only log significant positive rewards (food found)
            print(f"    💰 BIG REWARD: {reward:.2f}")
        
        # Return in gym format: obs, reward, terminated, truncated, info
        observation = self.get_state_array()
        terminated = self.game_over  # Game ends when snake dies
        truncated = False  # We don't truncate episodes
        info = {
            'score': self.score, 
            'hunger': self.steps_since_food,
            'hunger_penalty': (self.steps_since_food / self.HUNGER_MAX) ** 2 if not ate_food else 0,
            'starved_to_death': self.steps_since_food >= self.MAX_STARVATION_STEPS if self.game_over else False,
            'exploration_reward': exploration_reward,
            'areas_explored': len(self.visited_positions),
            'total_reward': reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_current_food_count(self):
        """
        Calculate current number of foods based on training progress (curriculum learning)
        Start with 1 food, add 1 food every 10% of training progress
        """
        # Handle case where global_step_count not initialized yet (during __init__)
        if not hasattr(self, 'global_step_count'):
            return self.INITIAL_FOODS
            
        # Calculate progress as percentage (0.0 to 1.0)
        progress = self.global_step_count / self.TOTAL_TRAINING_STEPS
        
        # Add 1 food every 10% of training progress
        # 0-9%: 1 food, 10-19%: 2 foods, 20-29%: 3 foods, etc.
        foods_to_add = int(progress * 10)  # 0, 1, 2, 3, ..., 9
        current_foods = self.INITIAL_FOODS + foods_to_add
        
        # Cap at maximum foods
        return min(current_foods, self.FINAL_FOODS)
    
    def _update_step_counter(self):
        """Update global step counter for curriculum learning"""
        self.global_step_count += 1
        
        # Log curriculum progress every 250k steps (10% of 2.5M)
        if self.global_step_count % 250000 == 0:
            current_foods = self._get_current_food_count()
            progress_pct = (self.global_step_count / self.TOTAL_TRAINING_STEPS) * 100
            print(f"🎓 CURRICULUM UPDATE: Step {self.global_step_count:,} ({progress_pct:.0f}%) - Now using {current_foods} foods")
    
    def _calculate_distance_reward(self):
        """
        Calculate reward based on distance to nearest food
        +DISTANCE_REWARD for each step closer, +DISTANCE_PENALTY for each step farther
        """
        # Get current distance to nearest food (Manhattan distance)
        current_distance = self._get_nearest_food_distance()
        
        # If this is the first step, just store distance
        if self.prev_distance_to_food is None:
            self.prev_distance_to_food = current_distance  
            return 0.0
        
        # Calculate distance change
        distance_change = self.prev_distance_to_food - current_distance
        
        # Reward based on getting closer/farther
        if distance_change > 0:  # Got closer
            distance_reward = distance_change * self.DISTANCE_REWARD
        elif distance_change < 0:  # Got farther  
            distance_reward = distance_change * self.DISTANCE_PENALTY  # This will be negative
        else:  # Same distance
            distance_reward = 0.0
        
        # Update for next step
        self.prev_distance_to_food = current_distance
        
        return distance_reward
    
    def _calculate_exploration_reward(self):
        """
        Calculate reward for exploration - encourage visiting new areas
        Higher reward for first visit, decaying reward for revisits
        """
        # Get current snake head position
        head_pos = tuple(self.snake[0])  # Convert to tuple for dictionary key
        
        # Track visits to this position
        if head_pos not in self.visited_positions:
            self.visited_positions[head_pos] = 0
        
        self.visited_positions[head_pos] += 1
        visit_count = self.visited_positions[head_pos]
        
        # Calculate exploration reward (decays with repeated visits)
        if visit_count == 1:
            # First visit gets full reward
            exploration_reward = self.EXPLORATION_REWARD
            # Only log occasionally for new areas (every 10th new area)
            if len(self.visited_positions) % 10 == 0:
                print(f"    🔍 EXPLORED {len(self.visited_positions)} areas (Score: {self.score})")
        else:
            # Repeated visits get decaying reward
            exploration_reward = self.EXPLORATION_REWARD * (self.EXPLORATION_DECAY ** (visit_count - 1))
        
        return exploration_reward
    
    def _place_multiple_foods(self):
        """Place multiple food items on the board (curriculum learning - dynamic count)"""
        # 🎓 Get current food count based on training progress
        target_food_count = self._get_current_food_count()
        
        self.foods = []  # List of food positions
        for _ in range(target_food_count):
            food_pos = self._place_single_food()
            if food_pos:  # If valid position found
                self.foods.append(food_pos)
        
        # Update parent's food reference to first food for compatibility
        self.food = self.foods[0] if self.foods else (0, 0)
        
        # Log food count changes and current status
        if hasattr(self, '_last_food_count') and self._last_food_count != target_food_count:
            print(f"🎓 Food count changed: {self._last_food_count} → {target_food_count} (step {self.global_step_count:,})")
        
        # Show current food count occasionally (every 100 episodes)
        if not hasattr(self, '_episode_count'):
            self._episode_count = 0
        self._episode_count += 1
        
        if self._episode_count % 100 == 0:
            progress = (self.global_step_count / self.TOTAL_TRAINING_STEPS) * 100
            print(f"🍎 Episode {self._episode_count}: {target_food_count} foods available (Progress: {progress:.1f}%)")
        
        self._last_food_count = target_food_count
    
    def _place_single_food(self):
        """Place a single food item at a random location"""
        max_attempts = 100  # Prevent infinite loop
        for _ in range(max_attempts):
            food_pos = (
                random.randint(0, (self.WINDOW_WIDTH - self.CELL_SIZE) // self.CELL_SIZE) * self.CELL_SIZE,
                random.randint(0, (self.WINDOW_HEIGHT - self.CELL_SIZE) // self.CELL_SIZE) * self.CELL_SIZE
            )
            # Make sure food doesn't spawn on snake or other foods
            if food_pos not in self.snake and food_pos not in self.foods:
                return food_pos
        return None  # Could not find valid position
    
    def _move_snake(self):
        """Override parent's move_snake to handle multiple foods with curriculum learning"""
        head_x, head_y = self.snake[0]
        d_x, d_y = self.direction
        new_head = (head_x + d_x, head_y + d_y)
        self.snake.insert(0, new_head)
        
        # Check if any food eaten
        if new_head in self.foods:
            self.score += 1
            # Remove eaten food
            self.foods.remove(new_head)
            
            # 🎓 CURRICULUM LEARNING: Adjust food count based on current progress
            target_food_count = self._get_current_food_count()
            
            # Add new food only if we haven't reached the target count
            if len(self.foods) < target_food_count:
                new_food = self._place_single_food()
                if new_food:
                    self.foods.append(new_food)
            
            # Update parent's food reference
            self.food = self.foods[0] if self.foods else (0, 0)
            return True
        else:
            self.snake.pop()
            return False
    
    def _get_nearest_food_distance(self):
        """Get distance to the nearest food"""
        if not self.foods:
            return float('inf')
        
        head_x, head_y = self.snake[0]
        min_distance = float('inf')
        
        for food_x, food_y in self.foods:
            distance = abs(head_x - food_x) + abs(head_y - food_y)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def get_state_array(self):
        """
        Convert game state to NumPy array for AI training
        Returns a flattened grid where:
        - 0 = empty space
        - 1 = snake head  
        - 0.5 = snake body
        - -1 = food
        """
        # Calculate grid dimensions (divide screen by cell size)
        grid_height = self.WINDOW_HEIGHT // self.CELL_SIZE  # 480 // 20 = 24
        grid_width = self.WINDOW_WIDTH // self.CELL_SIZE    # 640 // 20 = 32
        
        # Create empty grid filled with zeros
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # Mark snake positions
        for i, segment in enumerate(self.snake):
            # Convert pixel coordinates to grid coordinates
            x, y = segment[0] // self.CELL_SIZE, segment[1] // self.CELL_SIZE
            
            # Make sure coordinates are within grid bounds
            if 0 <= x < grid_width and 0 <= y < grid_height:
                if i == 0:
                    grid[y, x] = 1.0    # Head = 1
                else:
                    grid[y, x] = 0.5    # Body = 0.5
        
        # Mark all food positions
        for food_pos in self.foods:
            food_x, food_y = food_pos[0] // self.CELL_SIZE, food_pos[1] // self.CELL_SIZE
            if 0 <= food_x < grid_width and 0 <= food_y < grid_height:
                grid[food_y, food_x] = -1.0  # Food = -1
        
        # Flatten to 1D array (AI networks expect 1D input)
        return grid.flatten()
    
    def render(self):
        """Override parent's render to show multiple foods"""
        if not self.rendering:
            self.init_pygame()
        
        # Handle pygame events
        import pygame
        import sys
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.window.fill((0, 0, 0))
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0), (*segment, self.CELL_SIZE, self.CELL_SIZE))
        
        # Draw all foods
        for food_pos in self.foods:
            pygame.draw.rect(self.window, (255, 0, 0), (*food_pos, self.CELL_SIZE, self.CELL_SIZE))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        
        # Draw food count
        food_count_text = self.font.render(f"Foods: {len(self.foods)}", True, (255, 255, 255))
        self.window.blit(food_count_text, (10, 40))
        
        # Draw hunger level
        hunger_text = self.font.render(f"Hunger: {self.steps_since_food}/{self.MAX_STARVATION_STEPS}", True, (255, 255, 255))
        self.window.blit(hunger_text, (10, 70))
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render("Game Over! Press R to restart", True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(self.WINDOW_WIDTH//2, self.WINDOW_HEIGHT//2))
            self.window.blit(game_over_text, text_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.FPS)

# Test our basic environment
if __name__ == "__main__":
    env = SnakeEnvironment()
    
    print("Testing AI-ready environment...")
    
    # Test the gym-compatible interface
    obs, info = env.reset()
    print(f"✅ Reset successful!")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation type: {type(obs)}")
    print(f"   Info: {info}")
    
    # Test a few actions
    print(f"\n🎮 Testing actions...")
    for i in range(3):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"   Step {i+1}: {action_names[action]} → Reward: {reward}, Score: {info['score']}")
        
        if terminated:
            print("   🎯 Game over!")
            break
    
    print(f"\n🧠 ENVIRONMENT READY FOR AI TRAINING!")
    print(f"   Run 'python train_ai.py' to start training! 🚀")