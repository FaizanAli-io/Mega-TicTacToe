import torch
import pygame
from agent import DQNAgent
from game import TicTacToe7x7

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 700, 700  # Window size
GRID_SIZE = 7  # 7x7 grid
CELL_SIZE = WIDTH // GRID_SIZE
BG_COLOR = (255, 255, 255)  # White
LINE_COLOR = (0, 0, 0)  # Black
X_COLOR = (255, 0, 0)  # Red
O_COLOR = (0, 0, 255)  # Blue

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("7x7 Tic-Tac-Toe")


# ----------------------
# Pygame Visualizer Functions
# ----------------------
def draw_board(board):
    screen.fill(BG_COLOR)
    # Draw grid lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(
            screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT), 2
        )
        pygame.draw.line(
            screen, LINE_COLOR, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), 2
        )
    # Draw X's and O's
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row, col] == 1:  # X
                pygame.draw.line(
                    screen,
                    X_COLOR,
                    (col * CELL_SIZE + 10, row * CELL_SIZE + 10),
                    ((col + 1) * CELL_SIZE - 10, (row + 1) * CELL_SIZE - 10),
                    3,
                )
                pygame.draw.line(
                    screen,
                    X_COLOR,
                    ((col + 1) * CELL_SIZE - 10, row * CELL_SIZE + 10),
                    (col * CELL_SIZE + 10, (row + 1) * CELL_SIZE - 10),
                    3,
                )
            elif board[row, col] == -1:  # O
                pygame.draw.circle(
                    screen,
                    O_COLOR,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        row * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 2 - 10,
                    3,
                )


def get_mouse_cell():
    x, y = pygame.mouse.get_pos()
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row, col


def display_message(message):
    font = pygame.font.SysFont(None, 55)
    text = font.render(message, True, (0, 0, 0))
    screen.blit(
        text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2)
    )
    pygame.display.update()
    pygame.time.wait(2000)  # Wait 2 seconds before resetting


# ----------------------
# Load the Trained Model
# ----------------------
def load_model(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.epsilon = checkpoint["epsilon"]  # Set epsilon to a low value for testing
    print(f"Model loaded from {checkpoint_path}")


# Initialize the environment and agent
env = TicTacToe7x7()
agent = DQNAgent()

# Load the saved model
checkpoint_path = "saved_models/checkpoint_episode_200.pt"  # Path to your saved model
load_model(agent, checkpoint_path)

# Set epsilon to 0 for deterministic play (no exploration)
agent.epsilon = 0.0


# ----------------------
# Play Against the AI with Pygame
# ----------------------
def play_against_ai_pygame(agent):
    env = TicTacToe7x7()
    state = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and not done:
                # Human's turn (O)
                row, col = get_mouse_cell()
                if (row, col) in [tuple(move) for move in env.get_valid_moves()]:
                    state, reward, done, _ = env.step((row, col))
                    draw_board(state)
                    pygame.display.update()

                    if done:
                        if env.winner == -1:
                            display_message("You win!")
                        else:
                            display_message("It's a draw!")
                        break

                    # AI's turn (X)
                    valid_moves = env.get_valid_moves()
                    ai_action = agent.act(state, valid_moves)
                    state, reward, done, _ = env.step(ai_action)
                    draw_board(state)
                    pygame.display.update()

                    if done:
                        if env.winner == 1:
                            display_message("AI wins!")
                        else:
                            display_message("It's a draw!")
                        break

        draw_board(state)
        pygame.display.update()
        clock.tick(30)  # Limit to 30 FPS


# Start the game
play_against_ai_pygame(agent)
