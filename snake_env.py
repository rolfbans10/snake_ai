from snake_game import SnakeGame

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
        # For now, just return the game state dictionary
        return self.get_state()
    
    def step(self, action):
        """
        Take one step in the environment
        action: 0=up, 1=right, 2=down, 3=left
        """
        # Use parent's step method
        super().step(action)
        
        # Return the basic information an AI needs:
        # (state, reward, done, info)
        state = self.get_state()
        reward = 0  # We'll add rewards in the next step
        done = self.game_over
        info = {'score': self.score}
        
        return state, reward, done, info

# Test our basic environment
if __name__ == "__main__":
    env = SnakeEnvironment()
    
    print("Testing basic environment...")
    state = env.reset()
    print(f"Initial state keys: {state.keys()}")
    
    # Visual test - watch the snake randomly move around until it dies
    import random
    import time
    
    print("Watch the snake move randomly until it dies!")
    print("Close the window to stop early...")
    
    for i in range(1000):  # Try up to 1000 random actions
        action = random.randint(0, 3)  # Truly random actions
        state, reward, done, info = env.step(action)
        
        # Render the game so we can see it
        env.render()
        
        # Print current info
        snake_head = state['snake'][0]
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        print(f"Step {i+1}: {action_names[action]}, Head={snake_head}, Score={info['score']}")
        
        # Add a small delay so we can see the movement
        time.sleep(0.2)  # 0.2 seconds between moves
        
        if done:
            print("\n🎯 GAME OVER! Snake died!")
            print("Press any key to close...")
            # Keep the window open for a moment
            time.sleep(3)
            env.close()
            break
    
    if not done:
        print("Wow! Snake survived 1000 random moves!")
        env.close()