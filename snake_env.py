from snake_game import SnakeGame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeEnvironment(SnakeGame, gym.Env):
    """
    AI-Ready Snake Environment that extends SnakeGame and implements OpenAI Gym interface
    """
    
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
        
        # Return observation and info (gym format)
        observation = self.get_state_array()
        info = {'score': self.score}
        return observation, info
    
    def step(self, action):
        """
        Take one step in the environment
        action: 0=up, 1=right, 2=down, 3=left
        """
        # Use parent's step method
        super().step(action)
        
        # STEP 1, 2 & 3: Calculate reward based on what happened
        reward = 0
        
        # STEP 3: Step penalty - small cost for each move (encourages efficiency)
        reward -= 0.1
        
        # STEP 2: Food reward - check if score increased (ate food)
        if hasattr(self, 'prev_score'):
            if self.score > self.prev_score:
                reward += 10
        
        # Store current score for next time
        self.prev_score = self.score
        
        # STEP 1: Death penalty - overrides everything if we died
        if self.game_over:
            reward = -10  # Override any other rewards if we died
        
        # Return in gym format: obs, reward, terminated, truncated, info
        observation = self.get_state_array()
        terminated = self.game_over  # Game ends when snake dies
        truncated = False  # We don't truncate episodes
        info = {'score': self.score}
        
        return observation, reward, terminated, truncated, info
    
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