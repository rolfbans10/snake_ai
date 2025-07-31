from snake_game import SnakeGame
import numpy as np

class SnakeEnvironment(SnakeGame):
    """
    Step 1: Create a basic training environment that extends SnakeGame
    """
    
    def __init__(self):
        # Call the parent class constructor
        super().__init__()
    
    def reset(self):
        """Reset the game and return the initial state"""
        super().reset()  # Use parent's reset method
        # Initialize score tracking for food rewards
        self.prev_score = 0
        return self.get_state()
    
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
                print(f"    🍎 FOOD REWARD: +10 (ate food! Score: {self.score})")
        
        # Store current score for next time
        self.prev_score = self.score
        
        # STEP 1: Death penalty - overrides everything if we died
        if self.game_over:
            reward = -10  # Override any other rewards if we died
            print(f"    💀 DEATH PENALTY: -10 (snake died!)")
        
        # Return the basic information an AI needs:
        state = self.get_state()
        done = self.game_over
        info = {'score': self.score}
        
        return state, reward, done, info
    
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
        print(f"    🔢 Creating {grid_height}x{grid_width} grid...")
        
        # Create empty grid filled with zeros
        grid = np.zeros((grid_height, grid_width))
        
        # STEP 2A: Mark snake positions
        for i, segment in enumerate(self.snake):
            # Convert pixel coordinates to grid coordinates
            x, y = segment[0] // self.CELL_SIZE, segment[1] // self.CELL_SIZE
            
            # Make sure coordinates are within grid bounds
            if 0 <= x < grid_width and 0 <= y < grid_height:
                if i == 0:
                    grid[y, x] = 1    # Head = 1
                    print(f"    🐍 Snake head at grid ({x}, {y})")
                else:
                    grid[y, x] = 0.5  # Body = 0.5
        
        # STEP 2B: Mark food position  
        food_x, food_y = self.food[0] // self.CELL_SIZE, self.food[1] // self.CELL_SIZE
        if 0 <= food_x < grid_width and 0 <= food_y < grid_height:
            grid[food_y, food_x] = -1  # Food = -1
            print(f"    🍎 Food at grid ({food_x}, {food_y})")
        
        # STEP 2C: Flatten to 1D array (AI networks expect 1D input)
        flattened = grid.flatten()
        print(f"    📊 Grid flattened to {len(flattened)} numbers")
        
        return flattened

# Test our basic environment
if __name__ == "__main__":
    env = SnakeEnvironment()
    
    print("Testing NumPy array conversion...")
    
    # Reset and get initial state
    dict_state = env.reset()
    print(f"📝 Dictionary state keys: {dict_state.keys()}")
    
    # Test our new NumPy conversion
    print("\n🔢 CONVERTING TO NUMPY ARRAY:")
    array_state = env.get_state_array()
    
    print(f"\n📊 NUMPY ARRAY INFO:")
    print(f"   Shape: {array_state.shape}")
    print(f"   Type: {type(array_state)}")
    print(f"   Min value: {array_state.min()}")
    print(f"   Max value: {array_state.max()}")
    print(f"   Unique values: {np.unique(array_state)}")
    
    # Show a small sample of the array
    print(f"\n🔍 FIRST 20 VALUES:")
    print(f"   {array_state[:20]}")
    
    print(f"\n🧮 COMPARISON:")
    print(f"   Dictionary state: Complex objects (lists, tuples)")
    print(f"   NumPy array: {len(array_state)} simple numbers")
    print(f"   Perfect for AI neural networks! 🧠")