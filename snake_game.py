import pygame
import sys
import random

# Constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
FPS = 10
CELL_SIZE = 20

# staarting direction
direction = (CELL_SIZE, 0)


def move_snake(snake, direction, food):
    head_x, head_y = snake[0]
    d_x, d_y = direction
    new_head = (head_x + d_x, head_y + d_y)
    snake.insert(0, new_head)
    if new_head == food:
        return True
    else:
        snake.pop()
        return False

def handle_input(current_direction):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and current_direction != (0, CELL_SIZE):
        return (0, -CELL_SIZE)
    elif keys[pygame.K_DOWN] and current_direction != (0, -CELL_SIZE):
        return (0, CELL_SIZE)
    elif keys[pygame.K_LEFT] and current_direction != (CELL_SIZE, 0):
        return (-CELL_SIZE, 0)
    elif keys[pygame.K_RIGHT] and current_direction != (-CELL_SIZE, 0):
       return (CELL_SIZE, 0)
    return current_direction

def place_food():
    return (
        random.randint(0, (WINDOW_WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE,
        random.randint(0, (WINDOW_HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
    )

snake = [
    (100, 100),
    (80, 100),
    (60, 100)
]

food = place_food()


#initialize pygame
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()


#main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    direction = handle_input(direction)
    ate_food = move_snake(snake, direction, food)
    if ate_food:
        food = place_food()

    #draw the game
    window.fill((0, 0, 0))


    #draw the snake
    for segment in snake:
        pygame.draw.rect(window, (0, 255, 0), (*segment, CELL_SIZE, CELL_SIZE))

    # draw the food
    pygame.draw.rect(window, (255,0,0), (*food, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(FPS)