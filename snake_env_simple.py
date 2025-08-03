from snake_game import SnakeGame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleSnakeEnvironment(SnakeGame, gym.Env):
    """
    Simple Snake Environment with basic reward system
    """
    
    # Simple reward system
    INITIAL_BALANCE = 1    # Starting balance
    FOOD_REWARD = 10        # +10 for eating food
    DEATH_PENALTY = -1      # -1 for dying
    STEP_PENALTY = -0.01     # -0.1 each step
    DIRECTION_REWARD = 0.5   # +0.5 for moving toward food
    DIRECTION_PENALTY = -0.001 # -0.5 for moving away from food
    
    def __init__(self):
        # Set grid dimensions first (needed before parent init)
        self.WINDOW_WIDTH = 640
        self.WINDOW_HEIGHT = 480
        self.CELL_SIZE = 20
        self.grid_height = self.WINDOW_HEIGHT // self.CELL_SIZE  # 24
        self.grid_width = self.WINDOW_WIDTH // self.CELL_SIZE    # 32
        
        # Call parent constructor with our dimensions
        super().__init__(width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT, cell_size=self.CELL_SIZE)
        
        # Define action space: 4 actions (up, right, down, left)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        # Board (24x32=768) + reward(1) + direction(1) + food_quadrant(1) + distance(1) + score(1) + danger(4) = 777
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0, 
            shape=(777,),
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.current_reward_balance = self.INITIAL_BALANCE
        self.episode_step_count = 0
        self.prev_food_distance = None  # Track previous distance to food for directional rewards
        
        print(f"🐍 Simple Snake Environment created!")
        print(f"   Grid size: {self.grid_width}x{self.grid_height}")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space}")
        print(f"   Rewards: +{self.FOOD_REWARD} food, {self.DEATH_PENALTY} death, {self.STEP_PENALTY} step, ±{self.DIRECTION_REWARD} direction")
        print(f"   Features: Board(768) + Reward(1) + Direction(1) + FoodQuad(1) + Distance(1) + Score(1) + Dangers(4)")
    
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
        
        # Step penalty
        step_reward += self.STEP_PENALTY
        self.episode_step_count += 1
        
        # Food reward (if score increased)
        if self.score > prev_score:
            step_reward += self.FOOD_REWARD
            print(f"    🍎 Food eaten! Score: {self.score}, Reward: +{self.FOOD_REWARD}")
        
        # Directional reward (encourage moving toward food)
        directional_reward = self._calculate_directional_reward()
        step_reward += directional_reward
        
        # Death penalty
        if self.game_over:
            step_reward += self.DEATH_PENALTY
            # Update reward balance first to get final total
            self.current_reward_balance += step_reward
            print(f"    💀 Game over! Final score: {self.score}, Steps: {self.episode_step_count}, Final balance: {self.current_reward_balance:.2f}, Death penalty: {self.DEATH_PENALTY}")
        else:
            # Update reward balance for non-death steps
            self.current_reward_balance += step_reward
        
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
            'directional_reward': directional_reward
        }
        
        return observation, step_reward, terminated, truncated, info
    
    def get_observation(self):
        """Get current observation"""
        # 1. Board as 2D array with walls
        board = self.get_board_array()
        
        # 2. Current reward balance
        reward_value = [self.current_reward_balance]
        
        # 3. Current direction (encoded as 0-3)
        direction_value = [self.get_direction_encoding()]
        
        # 4. Food position (quadrant: 0=top_left, 1=top_right, 2=bottom_left, 3=bottom_right)
        food_quadrant = [self.get_food_quadrant()]
        
        # 5. Distance to food
        distance = [self.get_distance_to_food()]
        
        # 6. Current score
        score_value = [float(self.score)]
        
        # 7. Danger indicators (NEW - helps avoid immediate death)
        danger_indicators = self.get_danger_indicators()
        
        # Combine all observations
        observation = np.concatenate([
            board.flatten(),        # 768 values
            reward_value,          # 1 value
            direction_value,       # 1 value
            food_quadrant,         # 1 value
            distance,              # 1 value
            score_value,           # 1 value
            danger_indicators      # 4 values (NEW)
        ], dtype=np.float32)
        
        return observation
    
    def get_board_array(self):
        """Get 2D board representation with walls"""
        # Create board with walls (1=wall, 0=empty, 0.5=snake_body, 1=snake_head, -1=food)
        board = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Add walls around the border
        board[0, :] = -1.0      # Top wall
        board[-1, :] = -1.0     # Bottom wall  
        board[:, 0] = -1.0      # Left wall
        board[:, -1] = -1.0     # Right wall
        
        # Add snake
        for i, segment in enumerate(self.snake):
            x = segment[0] // self.CELL_SIZE
            y = segment[1] // self.CELL_SIZE
            
            # Make sure coordinates are within bounds (excluding walls)
            if 1 <= x < self.grid_width-1 and 1 <= y < self.grid_height-1:
                if i == 0:
                    board[y, x] = 1.0    # Head
                else:
                    board[y, x] = 0.5    # Body
        
        # Add food
        if self.food:
            food_x = self.food[0] // self.CELL_SIZE
            food_y = self.food[1] // self.CELL_SIZE
            if 1 <= food_x < self.grid_width-1 and 1 <= food_y < self.grid_height-1:
                board[food_y, food_x] = 1.0

        return board
    
    def get_direction_encoding(self):
        """Encode current direction as 0-3"""
        if self.direction == (0, -self.CELL_SIZE):     # up
            return 0
        elif self.direction == (self.CELL_SIZE, 0):    # right
            return 1
        elif self.direction == (0, self.CELL_SIZE):    # down
            return 2
        elif self.direction == (-self.CELL_SIZE, 0):   # left
            return 3
        else:
            return 0  # default
    
    def get_food_quadrant(self):
        """Get food position relative to snake head (0=top_left, 1=top_right, 2=bottom_left, 3=bottom_right)"""
        if not self.food or not self.snake:
            return 0
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Determine quadrant
        if food_x <= head_x and food_y <= head_y:
            return 0  # top_left
        elif food_x > head_x and food_y <= head_y:
            return 1  # top_right
        elif food_x <= head_x and food_y > head_y:
            return 2  # bottom_left
        else:
            return 3  # bottom_right
    
    def get_distance_to_food(self):
        """Get Manhattan distance to food"""
        if not self.food or not self.snake:
            return 0
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        return distance / self.CELL_SIZE  # Normalize by cell size
    
    def _calculate_directional_reward(self):
        """Calculate reward for moving toward/away from food"""
        current_distance = self.get_distance_to_food()
        
        # If this is the first step of episode, just store distance
        if self.prev_food_distance is None:
            self.prev_food_distance = current_distance
            return 0.0
        
        # Calculate distance change
        distance_change = self.prev_food_distance - current_distance
        
        # Give reward/penalty based on direction
        if distance_change > 0:  # Got closer to food
            directional_reward = distance_change * self.DIRECTION_REWARD
        elif distance_change < 0:  # Got farther from food
            directional_reward = distance_change * abs(self.DIRECTION_PENALTY)  # This will be negative
        else:  # Same distance
            directional_reward = 0.0
        
        # Update for next step
        self.prev_food_distance = current_distance
        
        return directional_reward
    
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

# Test the simple environment
if __name__ == "__main__":
    env = SimpleSnakeEnvironment()
    
    print("\n🧪 Testing Simple Snake Environment...")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful!")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Initial info: {info}")
    
    # Test a few steps
    print(f"\n🎮 Testing actions...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"   Step {i+1}: {action_names[action]} → Reward: {reward:.2f}, Balance: {info['reward_balance']:.2f}, Score: {info['score']}")
        
        if terminated:
            print("   🎯 Game over!")
            break
    
    print(f"\n🚀 Simple environment ready for training!")