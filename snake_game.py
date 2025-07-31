import pygame
import sys
import random

class SnakeGame:
    def __init__(self, width=640, height=480, cell_size=20, fps=10):
        # Game constants
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height
        self.CELL_SIZE = cell_size
        self.FPS = fps
        
        # Game state
        self.snake = None
        self.food = None
        self.direction = None
        self.score = 0
        self.game_over = False
        
        # Pygame setup (optional for rendering)
        self.window = None
        self.clock = None
        self.font = None
        self.rendering = False
        
        # Initialize game
        self.reset()
    
    def init_pygame(self):
        """Initialize pygame for rendering"""
        if not self.rendering:
            pygame.init()
            self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)
            self.rendering = True
    
    def reset(self):
        """Reset the game to initial state"""
        self.snake = [
            (100, 100),
            (80, 100),
            (60, 100)
        ]
        self.direction = (self.CELL_SIZE, 0)  # Moving right
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
    
    def _place_food(self):
        """Place food at a random location"""
        while True:
            food_pos = (
                random.randint(0, (self.WINDOW_WIDTH - self.CELL_SIZE) // self.CELL_SIZE) * self.CELL_SIZE,
                random.randint(0, (self.WINDOW_HEIGHT - self.CELL_SIZE) // self.CELL_SIZE) * self.CELL_SIZE
            )
            # Make sure food doesn't spawn on snake
            if food_pos not in self.snake:
                return food_pos
    
    def _move_snake(self):
        """Move the snake in current direction"""
        head_x, head_y = self.snake[0]
        d_x, d_y = self.direction
        new_head = (head_x + d_x, head_y + d_y)
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            return True
        else:
            self.snake.pop()
            return False
    
    def _check_collision(self):
        """Check for wall or self collision"""
        head = self.snake[0]
        
        # Wall collision
        if (head[0] < 0 or head[0] >= self.WINDOW_WIDTH or
            head[1] < 0 or head[1] >= self.WINDOW_HEIGHT):
            return True
        
        # Self collision
        if head in self.snake[1:]:
            return True
        
        return False
    
    def step(self, action=None):
        """
        Take one game step
        action: 0=up, 1=right, 2=down, 3=left, None=continue current direction
        """
        if self.game_over:
            return
        
        # Update direction based on action
        if action is not None:
            self._set_direction_from_action(action)
        
        # Move snake
        self._move_snake()
        
        # Check collision
        if self._check_collision():
            self.game_over = True
    
    def _set_direction_from_action(self, action):
        """Convert action to direction, preventing reverse moves"""
        current_dir = self.direction
        
        if action == 0:  # Up
            new_dir = (0, -self.CELL_SIZE)
        elif action == 1:  # Right
            new_dir = (self.CELL_SIZE, 0)
        elif action == 2:  # Down
            new_dir = (0, self.CELL_SIZE)
        elif action == 3:  # Left
            new_dir = (-self.CELL_SIZE, 0)
        else:
            return  # Invalid action, keep current direction
        
        # Prevent snake from reversing into itself
        if new_dir != (-current_dir[0], -current_dir[1]):
            self.direction = new_dir
    
    def handle_pygame_input(self):
        """Handle pygame keyboard input"""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.direction != (0, self.CELL_SIZE):
            self.direction = (0, -self.CELL_SIZE)
        elif keys[pygame.K_DOWN] and self.direction != (0, -self.CELL_SIZE):
            self.direction = (0, self.CELL_SIZE)
        elif keys[pygame.K_LEFT] and self.direction != (self.CELL_SIZE, 0):
            self.direction = (-self.CELL_SIZE, 0)
        elif keys[pygame.K_RIGHT] and self.direction != (-self.CELL_SIZE, 0):
            self.direction = (self.CELL_SIZE, 0)
    
    def get_state(self):
        """Get current game state"""
        return {
            'snake': self.snake.copy(),
            'food': self.food,
            'direction': self.direction,
            'score': self.score,
            'game_over': self.game_over
        }
    
    def render(self):
        """Render the game using pygame"""
        if not self.rendering:
            self.init_pygame()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.window.fill((0, 0, 0))
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0), (*segment, self.CELL_SIZE, self.CELL_SIZE))
        
        # Draw food
        pygame.draw.rect(self.window, (255, 0, 0), (*self.food, self.CELL_SIZE, self.CELL_SIZE))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render("Game Over! Press R to restart", True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(self.WINDOW_WIDTH//2, self.WINDOW_HEIGHT//2))
            self.window.blit(game_over_text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(self.FPS)
    
    def close(self):
        """Clean up pygame"""
        if self.rendering:
            pygame.quit()

def play_human():
    """Play the game with human keyboard input"""
    game = SnakeGame()
    
    while True:
        game.render()
        
        if not game.game_over:
            game.handle_pygame_input()
            game.step()
        else:
            # Check for restart
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                game.reset()

if __name__ == "__main__":
    play_human()