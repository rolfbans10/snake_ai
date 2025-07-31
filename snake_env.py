from snake_game import SnakeGame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeEnvironment(SnakeGame, gym.Env):
    """
    AI-Ready Snake Environment that extends SnakeGame and implements OpenAI Gym interface
    """
    
    # 🎯 CONFIGURABLE REWARD SYSTEM
    DEATH_PENALTY = -100        # Penalty for dying
    FOOD_REWARD = 150          # Reward for eating food  
    STEP_PENALTY = -1      # Small penalty each step
    HUNGER_MAX = 20          # Max steps before hunger penalty maxes out
    DISTANCE_REWARD = 5     # Reward for moving closer to food
    DISTANCE_PENALTY = -10   # Penalty for moving away from food
    MAX_STARVATION_STEPS = 100  # Force death after this many steps without food
    
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
        
        print(f"🤖 AI Environment created!")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space}")
    
    def reset(self, seed=None):
        """Reset environment for AI training"""
        super().reset()  # Use parent's reset method
        self.prev_score = 0
        
        # HUNGER SYSTEM: Track steps since last food
        self.steps_since_food = 0
        
        # DISTANCE TRACKING: Track distance to food for directional rewards
        self.prev_distance_to_food = None
        
        # Return observation and info (gym format)
        observation = self.get_state_array()
        info = {'score': self.score, 'hunger': self.steps_since_food}
        return observation, info
    
    def step(self, action):
        """
        Take one step in the environment
        action: 0=up, 1=right, 2=down, 3=left
        """
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
                print(f"    🍎 FOOD FOUND! Hunger reset. +{self.FOOD_REWARD} reward")
        
        # Store current score for next time
        self.prev_score = self.score
        
        # STEP 3: DISTANCE REWARD - encourage moving toward food
        distance_reward = self._calculate_distance_reward()
        reward += distance_reward
        
        # STEP 4: STEP PENALTY - small penalty for each step to encourage efficiency
        reward += self.STEP_PENALTY
        
        # STEP 5: HUNGER PENALTY - grows over time without food
        if not ate_food:
            # Progressive hunger penalty that gets worse over time
            hunger_penalty = (self.steps_since_food / self.HUNGER_MAX) ** 2
            reward -= hunger_penalty
            
            # Show hunger status occasionally
            if self.steps_since_food % 25 == 0:
                print(f"    😰 HUNGER: {self.steps_since_food} steps, penalty: -{hunger_penalty:.2f}")
        
        # STEP 6: STARVATION DEATH - force death if too many steps without food
        if self.steps_since_food >= self.MAX_STARVATION_STEPS:
            self.game_over = True  # Force death due to starvation
            print(f"    💀 STARVED TO DEATH! {self.steps_since_food} steps without food!")
        
        # STEP 7: Death penalty - overrides everything if we died
        if self.game_over:
            reward = self.DEATH_PENALTY  # Override any other rewards if we died
            print(f"    💀 DEATH + STARVATION: {self.DEATH_PENALTY + (self.steps_since_food/10):.1f}")
        
        # Return in gym format: obs, reward, terminated, truncated, info
        observation = self.get_state_array()
        terminated = self.game_over  # Game ends when snake dies
        truncated = False  # We don't truncate episodes
        info = {
            'score': self.score, 
            'hunger': self.steps_since_food,
            'hunger_penalty': (self.steps_since_food / self.HUNGER_MAX) ** 2 if not ate_food else 0,
            'starved_to_death': self.steps_since_food >= self.MAX_STARVATION_STEPS if self.game_over else False
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_distance_reward(self):
        """
        Calculate reward based on distance to food
        +DISTANCE_REWARD for each step closer, +DISTANCE_PENALTY for each step farther
        """
        # Get current distance to food (Manhattan distance)
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        current_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
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
        
        # Mark food position  
        food_x, food_y = self.food[0] // self.CELL_SIZE, self.food[1] // self.CELL_SIZE
        if 0 <= food_x < grid_width and 0 <= food_y < grid_height:
            grid[food_y, food_x] = -1.0  # Food = -1
        
        # Flatten to 1D array (AI networks expect 1D input)
        return grid.flatten()

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