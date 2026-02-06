from snake_game import SnakeGame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MinimalSnakeEnvironment(SnakeGame, gym.Env):
    """
    Minimal Snake Environment with focused feature set (no board representation)
    Uses only essential features for faster training and better performance
    """
    
    # Improved reward system
    INITIAL_BALANCE = 1       # Starting balance
    FOOD_REWARD = 10          # +10 for eating food
    SURVIVAL_REWARD = 0.1     # +0.1 for each step survived
    DIRECTION_REWARD = 0.2    # +0.2 per cell closer to food (gentle guidance)
    DIRECTION_PENALTY = 0.0   # No penalty for moving away (allows exploration)
    # No death penalty - natural consequences (episode ends, no more rewards)
    
    def __init__(self):
        # Set grid dimensions first (needed before parent init)
        self.WINDOW_WIDTH = 640
        self.WINDOW_HEIGHT = 480
        self.CELL_SIZE = 20
        self.grid_height = self.WINDOW_HEIGHT // self.CELL_SIZE  # 24
        self.grid_width = self.WINDOW_WIDTH // self.CELL_SIZE    # 32
        
        # Calculate max snake length for normalization (needed before parent init calls reset)
        self.max_snake_length = self.grid_width * self.grid_height
        
        # Call parent constructor with our dimensions
        super().__init__(width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT, cell_size=self.CELL_SIZE)
        
        # Define action space: 4 actions (up, right, down, left)
        self.action_space = spaces.Discrete(4)
        
        # Define IMPROVED observation space (NO BOARD!)
        # Features: dangers(4) + food_binary(4) + distance(1) + direction_onehot(4) + 
        #          safe_moves(1) + snake_length(1) + body_proximity(1) = 16
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0, 
            shape=(16,),  # Improved: 16 features, all properly normalized
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.current_reward_balance = self.INITIAL_BALANCE
        self.episode_step_count = 0
        self.prev_food_distance = None  # Track previous distance to food for directional rewards
        
        print(f"🐍 Improved Minimal Snake Environment created!")
        print(f"   Grid size: {self.grid_width}x{self.grid_height}")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space} (16 features, all normalized 0-1)")
        print(f"   Rewards: +{self.FOOD_REWARD} food, +{self.SURVIVAL_REWARD} survive, +{self.DIRECTION_REWARD}/cell toward food")
        print(f"   Features: Dangers(4) + FoodBinary(4) + Distance(1) + DirectionOneHot(4) + SafeMoves(1) + Length(1) + BodyProx(1) = 16")
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset()
        
        # Reset tracking variables
        self.current_reward_balance = self.INITIAL_BALANCE
        self.episode_step_count = 0
        self.prev_food_distance = None  # Reset distance tracking for new episode
        
        # Return observation and info
        observation = self.get_observation()
        info = {
            'score': self.score,
            'reward_balance': self.current_reward_balance,
            'step_count': self.episode_step_count
        }
        
        return observation, info
    
    def step(self, action):
        """Take one step in the environment"""
        # Map action to direction
        # 0=up, 1=right, 2=down, 3=left
        direction_map = {
            0: (0, -self.CELL_SIZE),   # up
            1: (self.CELL_SIZE, 0),    # right  
            2: (0, self.CELL_SIZE),    # down
            3: (-self.CELL_SIZE, 0)    # left
        }
        
        # Convert action to int (in case it's a numpy array from model)
        action = int(action)
        
        # Set direction (avoid immediate reversal)
        new_direction = direction_map[action]
        if (new_direction[0] + self.direction[0] != 0 or 
            new_direction[1] + self.direction[1] != 0):
            self.direction = new_direction
        
        # Store previous score to detect food eaten
        prev_score = self.score
        
        # Take a step in the base game
        super().step(action)
        
        # Calculate reward for this step
        step_reward = 0
        
        # Survival reward (positive for staying alive!)
        step_reward += self.SURVIVAL_REWARD
        self.episode_step_count += 1
        
        # Food reward (if score increased)
        if self.score > prev_score:
            step_reward += self.FOOD_REWARD
            print(f"    🍎 Food eaten! Score: {self.score}, Reward: +{self.FOOD_REWARD}")
        
        # Directional reward (encourage moving toward food)
        directional_reward = self._calculate_directional_reward()
        step_reward += directional_reward
        
        # Update reward balance
        self.current_reward_balance += step_reward
        
        # Log game over (no death penalty - natural consequences only)
        if self.game_over:
            print(f"    💀 Game over! Final score: {self.score}, Length: {len(self.snake)}, Steps: {self.episode_step_count}")
            print(f"       Final balance: {self.current_reward_balance:.2f} (no death penalty)")
        
        # Get observation
        observation = self.get_observation()
        
        # Prepare return values
        terminated = self.game_over
        truncated = False
        info = {
            'score': self.score,
            'reward_balance': self.current_reward_balance,
            'step_count': self.episode_step_count,
            'step_reward': step_reward,
            'directional_reward': directional_reward,
            'snake_length': len(self.snake)
        }
        
        return observation, step_reward, terminated, truncated, info
    
    def get_observation(self):
        """Get IMPROVED observation - 16 features, all normalized 0-1"""
        # 1. Danger indicators (4 values) - immediate collision detection
        danger_indicators = self.get_danger_indicators()
        
        # 2. Food direction binary (4 values) - is food up/right/down/left?
        food_binary = self.get_food_direction_binary()
        
        # 3. Distance to food (1 value) - normalized 0-1
        distance = [self.get_distance_to_food()]
        
        # 4. Current direction one-hot (4 values) - proper encoding
        direction_onehot = self.get_direction_onehot()
        
        # 5. Safe moves count (1 value) - normalized 0-1
        safe_moves = [self.get_safe_moves_count()]
        
        # 6. Snake length (1 value) - normalized 0-1
        snake_length = [len(self.snake) / self.max_snake_length]
        
        # 7. Body proximity (1 value) - normalized 0-1
        body_proximity = [self.get_body_proximity_normalized()]
        
        # Combine all observations
        observation = np.concatenate([
            danger_indicators,    # 4 values (0 or 1)
            food_binary,          # 4 values (0 or 1)
            distance,             # 1 value (0 to 1)
            direction_onehot,     # 4 values (one-hot)
            safe_moves,           # 1 value (0 to 1)
            snake_length,         # 1 value (0 to 1)
            body_proximity        # 1 value (0 to 1)
        ], dtype=np.float32)
        
        # Total: 16 values, all in 0-1 range
        return observation
    
    def get_danger_indicators(self):
        """
        Get danger indicators for each direction (up, right, down, left)
        Returns [up_danger, right_danger, down_danger, left_danger]
        1.0 = danger (wall or self collision), 0.0 = safe
        """
        if not self.snake:
            return [0.0, 0.0, 0.0, 0.0]
        
        head_x, head_y = self.snake[0]
        dangers = []
        
        # Check each direction: up, right, down, left (matching action space)
        directions = [
            (0, -self.CELL_SIZE),   # up (action 0)
            (self.CELL_SIZE, 0),    # right (action 1)  
            (0, self.CELL_SIZE),    # down (action 2)
            (-self.CELL_SIZE, 0)    # left (action 3)
        ]
        
        for dx, dy in directions:
            next_x, next_y = head_x + dx, head_y + dy
            
            # Check wall collision
            wall_danger = (next_x < 0 or next_x >= self.WINDOW_WIDTH or 
                          next_y < 0 or next_y >= self.WINDOW_HEIGHT)
            
            # Check self collision
            self_danger = (next_x, next_y) in self.snake
            
            # 1.0 if dangerous, 0.0 if safe
            dangers.append(1.0 if (wall_danger or self_danger) else 0.0)
        
        return dangers
    
    def get_food_direction_binary(self):
        """Get binary indicators for food direction [up, right, down, left]"""
        if not self.food or not self.snake:
            return [0.0, 0.0, 0.0, 0.0]
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Binary indicators for each direction
        food_up = 1.0 if food_y < head_y else 0.0
        food_right = 1.0 if food_x > head_x else 0.0
        food_down = 1.0 if food_y > head_y else 0.0
        food_left = 1.0 if food_x < head_x else 0.0
        
        return [food_up, food_right, food_down, food_left]
    
    def get_direction_onehot(self):
        """Get current direction as one-hot encoded [up, right, down, left]"""
        onehot = [0.0, 0.0, 0.0, 0.0]
        
        if self.direction == (0, -self.CELL_SIZE):     # up
            onehot[0] = 1.0
        elif self.direction == (self.CELL_SIZE, 0):    # right
            onehot[1] = 1.0
        elif self.direction == (0, self.CELL_SIZE):    # down
            onehot[2] = 1.0
        elif self.direction == (-self.CELL_SIZE, 0):   # left
            onehot[3] = 1.0
        
        return onehot
    
    def get_body_proximity_normalized(self):
        """Get normalized proximity to nearest body segment (0 = far, 1 = adjacent)"""
        if len(self.snake) <= 3:
            return 0.0  # No dangerous body segments
        
        head_x, head_y = self.snake[0]
        min_dist = float('inf')
        
        # Check all body segments except head and first 2 body segments
        for segment in self.snake[3:]:
            dist = abs(head_x - segment[0]) + abs(head_y - segment[1])
            min_dist = min(min_dist, dist)
        
        # Normalize: 1 cell = 1.0, farther = lower value
        # Max meaningful distance is ~10 cells
        if min_dist == float('inf'):
            return 0.0
        
        proximity = 1.0 - min(min_dist / self.CELL_SIZE / 10.0, 1.0)
        return proximity
    
    def get_distance_to_food(self):
        """Get normalized Manhattan distance to food"""
        if not self.food or not self.snake:
            return 0.0
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Normalize by maximum possible distance
        max_distance = self.WINDOW_WIDTH + self.WINDOW_HEIGHT
        return distance / max_distance
    
    def _calculate_directional_reward(self):
        """Calculate reward for moving toward/away from food using RAW distance (in cells)"""
        current_distance = self.get_raw_distance_to_food()  # Raw distance in cells
        
        # If this is the first step of episode, just store distance
        if self.prev_food_distance is None:
            self.prev_food_distance = current_distance
            return 0.0
        
        # Calculate distance change (positive = got closer, negative = got farther)
        distance_change = self.prev_food_distance - current_distance
        
        # Give reward/penalty based on direction
        # Each cell closer = +DIRECTION_REWARD, each cell farther = +DIRECTION_PENALTY
        if distance_change > 0:  # Got closer to food
            directional_reward = distance_change * self.DIRECTION_REWARD
        elif distance_change < 0:  # Got farther from food
            directional_reward = distance_change * abs(self.DIRECTION_PENALTY)  # This will be negative
        else:  # Same distance
            directional_reward = 0.0
        
        # Update for next step
        self.prev_food_distance = current_distance
        
        return directional_reward
    
    def get_raw_distance_to_food(self):
        """Get RAW Manhattan distance to food in cells (not normalized)"""
        if not self.food or not self.snake:
            return 0
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Manhattan distance in pixels, converted to cells
        distance_pixels = abs(head_x - food_x) + abs(head_y - food_y)
        distance_cells = distance_pixels / self.CELL_SIZE
        
        return distance_cells
    
    def get_safe_moves_count(self):
        """Get count of safe directions to move (0-1 normalized)"""
        dangers = self.get_danger_indicators()
        safe_count = sum(1 for d in dangers if d == 0.0)
        return safe_count / 4.0  # Normalize to 0-1 range
    
# Test the minimal environment
if __name__ == "__main__":
    env = MinimalSnakeEnvironment()
    
    print("\n🧪 Testing Minimal Snake Environment...")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful!")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Initial info: {info}")
    print(f"   Observation values: {obs}")
    
    # Test a few steps
    print(f"\n🎮 Testing actions...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"   Step {i+1}: {action_names[action]} → Reward: {reward:.2f}, Balance: {info['reward_balance']:.2f}, Score: {info['score']}")
        print(f"      Observation: {obs}")
        
        if terminated:
            print("   🎯 Game over!")
            break
    
    print(f"\n🚀 Improved minimal environment ready for training!")
    print(f"   🎯 16 features (all normalized 0-1) vs 777 - trains MUCH faster!")
    print(f"   ✨ Improved: One-hot direction, binary food direction, proper normalization")