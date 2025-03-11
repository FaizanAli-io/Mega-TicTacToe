import numpy as np


class TicTacToe7x7:
    def __init__(self):
        self.board = np.zeros((7, 7))
        self.current_player = 1
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((7, 7))
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def check_winner(self, player):
        board = self.board
        for i in range(7):
            for j in range(4):
                if np.all(board[i, j : j + 4] == player):
                    return True
                if np.all(board[j : j + 4, i] == player):
                    return True

        for i in range(4):
            for j in range(4):
                if np.all([board[i + k][j + k] == player for k in range(4)]):
                    return True

        for i in range(4):
            for j in range(3, 7):
                if np.all([board[i + k][j - k] == player for k in range(4)]):
                    return True
        return False

    def step(self, action):
        row, col = action
        if self.board[row, col] != 0:
            return self.board.copy(), -10, True, {}

        self.board[row, col] = self.current_player

        if self.check_winner(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1
        elif len(self.get_valid_moves()) == 0:
            self.done = True
            reward = 0
        else:
            reward = 0
            self.current_player *= -1

        return self.board.copy(), reward, self.done, {}
