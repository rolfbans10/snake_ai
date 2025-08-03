from snake_game import SnakeGame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MinimalSnakeEnvironment(SnakeGame, gym.Env):
    """
    Minimal Snake Environment with focused feature set (no board representation)
    Uses only essential features for faster training and better performance
    """
    
    # Simple reward system
    INITIAL_BALANCE = 1    # Starting balance
    FOOD_REWARD = 10        # +10 for eating food
    BASE_DEATH_PENALTY = -1  # Base death penalty (scales with length)
    STEP_PENALTY = -0.01     # -0.01 each step
    DIRECTION_REWARD = 0.5   # +0.5 for moving toward food
    DIRECTION_PENALTY = -0.5 # -0.5 for moving away from food
    
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
        
        # Define ENHANCED MINIMAL observation space (NO BOARD!)
        # Features: dangers(4) + food_direction(2) + wall_distances(4) + snake_length(1) + 
        #          distance_to_food(1) + current_direction(1) + body_proximity(1) + reward_balance(1) +
        #          safe_moves_count(1) + recent_moves(2) + tail_direction(2) = 20
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0, 
            shape=(20,),  # Enhanced: 20 vs 15 vs 777
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.current_reward_balance = self.INITIAL_BALANCE
        self.episode_step_count = 0
        self.prev_food_distance = None  # Track previous distance to food for directional rewards
        self.move_history = [0, 0]  # Track last 2 moves for pattern detection
        
        print(f"🐍 Enhanced Minimal Snake Environment created!")
        print(f"   Grid size: {self.grid_width}x{self.grid_height}")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space} (ENHANCED MINIMAL - no board!)")
        print(f"   Rewards: +{self.FOOD_REWARD} food, {self.BASE_DEATH_PENALTY}*length death (DYNAMIC!), {self.STEP_PENALTY} step, ±{self.DIRECTION_REWARD} direction")
        print(f"   Features: Dangers(4) + FoodDir(2) + Walls(4) + Length(1) + Distance(1) + Direction(1) + BodyProx(1) + Balance(1) + SafeMoves(1) + History(2) + Tail(2) = 20")
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset()
        
        # Reset tracking variables
        self.current_reward_balance = self.INITIAL_BALANCE
        self.episode_step_count = 0
        self.prev_food_distance = None  # Reset distance tracking for new episode
        self.move_history = [0, 0]  # Reset movement history for new episode
        
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
        
        # Update movement history (track last 2 moves)
        self.move_history[1] = self.move_history[0]  # Previous move becomes older
        self.move_history[0] = action  # Current move becomes most recent
        
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
        
        # Dynamic death penalty (scales with snake length)
        if self.game_over:
            death_penalty = self.get_dynamic_death_penalty()
            step_reward += death_penalty
            # Update reward balance first to get final total
            self.current_reward_balance += step_reward
            print(f"    💀 Game over! Final score: {self.score}, Length: {len(self.snake)}, Steps: {self.episode_step_count}")
            print(f"       Final balance: {self.current_reward_balance:.2f}, Dynamic death penalty: {death_penalty:.2f}")
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
            'directional_reward': directional_reward,
            'death_penalty': self.get_dynamic_death_penalty() if self.game_over else 0,
            'snake_length': len(self.snake)
        }
        
        return observation, step_reward, terminated, truncated, info
    
    def get_observation(self):
        """Get MINIMAL observation (NO BOARD!)"""
        # 1. Danger indicators (4 values) - immediate safety
        danger_indicators = self.get_danger_indicators()
        
        # 2. Food direction vector (2 values) - where is food relative to head
        food_direction = self.get_food_direction_vector()
        
        # 3. Wall distances (4 values) - spatial awareness
        wall_distances = self.get_wall_distances()
        
        # 4. Snake length (1 value) - growth tracking
        snake_length = [float(len(self.snake))]
        
        # 5. Distance to food (1 value) - how far to food
        distance = [self.get_distance_to_food()]
        
        # 6. Current direction (1 value) - movement state
        direction_value = [float(self.get_direction_encoding())]
        
        # 7. Body proximity (1 value) - collision risk
        body_proximity = [self.get_body_proximity()]
        
        # 8. Current reward balance (1 value) - performance tracking
        reward_value = [self.current_reward_balance]
        
        # 9. Safe moves count (1 value) - tactical awareness
        safe_moves = [self.get_safe_moves_count()]
        
        # 10. Recent moves (2 values) - pattern detection
        recent_moves = [float(self.move_history[0]), float(self.move_history[1])]
        
        # 11. Tail direction (2 values) - body management
        tail_direction = self.get_tail_direction()
        
        # Combine all observations into enhanced minimal feature set
        observation = np.concatenate([
            danger_indicators,    # 4 values
            food_direction,       # 2 values
            wall_distances,       # 4 values
            snake_length,         # 1 value
            distance,             # 1 value
            direction_value,      # 1 value
            body_proximity,       # 1 value
            reward_value,         # 1 value
            safe_moves,           # 1 value (NEW)
            recent_moves,         # 2 values (NEW)
            tail_direction        # 2 values (NEW)
        ], dtype=np.float32)
        
        # Total: 20 values (vs 15 before, vs 777 in full board version!)
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
    
    def get_food_direction_vector(self):
        """Get normalized direction vector to food (dx, dy)"""
        if not self.food or not self.snake:
            return [0.0, 0.0]
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Calculate direction vector
        dx = food_x - head_x
        dy = food_y - head_y
        
        # Normalize to -1 to 1 range
        max_dist = max(abs(dx), abs(dy), 1)  # Avoid division by zero
        
        return [dx / max_dist, dy / max_dist]
    
    def get_wall_distances(self):
        """Get distance to walls in each direction [up, right, down, left]"""
        if not self.snake:
            return [0.0, 0.0, 0.0, 0.0]
        
        head_x, head_y = self.snake[0]
        
        distances = [
            head_y,                                        # distance to top wall
            self.WINDOW_WIDTH - head_x - self.CELL_SIZE,   # distance to right wall  
            self.WINDOW_HEIGHT - head_y - self.CELL_SIZE,  # distance to bottom wall
            head_x                                         # distance to left wall
        ]
        
        # Normalize by cell size and max distance
        max_distance = max(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        return [d / max_distance for d in distances]
    
    def get_body_proximity(self):
        """Get distance to nearest body segment (excluding head and neck)"""
        if len(self.snake) <= 3:
            return 10.0  # No body segments close enough to matter
        
        head_x, head_y = self.snake[0]
        min_dist = float('inf')
        
        # Check all body segments except head and first 2 body segments
        for segment in self.snake[3:]:
            dist = abs(head_x - segment[0]) + abs(head_y - segment[1])
            min_dist = min(min_dist, dist)
        
        # Normalize and cap at reasonable value
        normalized_dist = min_dist / self.CELL_SIZE
        return min(normalized_dist, 10.0)  # Cap at 10 for reasonable range
    
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
    
    def get_safe_moves_count(self):
        """Get count of safe directions to move (0-1 normalized)"""
        dangers = self.get_danger_indicators()
        safe_count = sum(1 for d in dangers if d == 0.0)
        return safe_count / 4.0  # Normalize to 0-1 range
    
    def get_tail_direction(self):
        """Get normalized direction from head to tail"""
        if len(self.snake) < 2:
            return [0.0, 0.0]
        
        head_x, head_y = self.snake[0] 
        tail_x, tail_y = self.snake[-1]
        
        # Calculate direction vector from head to tail
        dx = tail_x - head_x
        dy = tail_y - head_y
        
        # Normalize by window dimensions to get -1 to 1 range
        dx_norm = dx / self.WINDOW_WIDTH
        dy_norm = dy / self.WINDOW_HEIGHT
        
        return [dx_norm, dy_norm]
    
    def get_dynamic_death_penalty(self):
        """
        Calculate death penalty based on snake length
        The longer the snake, the more severe the penalty for dying
        """
        snake_length = len(self.snake)
        initial_length = 3  # Starting snake length
        
        # Length-based multiplier: starts at 1.0, increases with growth
        # Formula: penalty = base * (current_length / initial_length)
        length_multiplier = snake_length / initial_length
        
        # Cap the maximum penalty to prevent excessive punishment
        max_multiplier = 5.0  # Maximum 5x penalty
        length_multiplier = min(length_multiplier, max_multiplier)
        
        dynamic_penalty = self.BASE_DEATH_PENALTY * length_multiplier
        
        return dynamic_penalty

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
    
    print(f"\n🚀 Enhanced minimal environment ready for training!")
    print(f"   🎯 20 features vs 777 - should train MUCH faster!")
    print(f"   🆕 NEW: Safe moves count, movement history, tail direction")