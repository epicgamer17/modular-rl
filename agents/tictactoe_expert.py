import numpy as np


class TicTacToeBestAgent:
    def __init__(self, name="tictactoe_expert"):
        self.name = name

    def predict(self, observation, info, env=None, *args, **kwargs):
        return observation, info

    def select_actions(self, prediction, info, *args, **kwargs):
        # Reconstruct board: +1 for current player, -1 for opponent, 0 otherwise
        board = prediction[0][0] - prediction[0][1]
        # print(board)
        # Default: random legal move
        action = np.random.choice(info["legal_moves"])

        # Horizontal and vertical checks
        for i in range(3):
            # Row
            if np.sum(board[i, :]) == 2 and 0 in board[i, :]:
                ind = np.where(board[i, :] == 0)[0][0]
                return np.ravel_multi_index((i, ind), (3, 3))
            elif abs(np.sum(board[i, :])) == 2 and 0 in board[i, :]:
                ind = np.where(board[i, :] == 0)[0][0]
                action = np.ravel_multi_index((i, ind), (3, 3))

            # Column
            if np.sum(board[:, i]) == 2 and 0 in board[:, i]:
                ind = np.where(board[:, i] == 0)[0][0]
                return np.ravel_multi_index((ind, i), (3, 3))
            elif abs(np.sum(board[:, i])) == 2 and 0 in board[:, i]:
                ind = np.where(board[:, i] == 0)[0][0]
                action = np.ravel_multi_index((ind, i), (3, 3))

        # Diagonals
        diag = board.diagonal()
        if np.sum(diag) == 2 and 0 in diag:
            ind = np.where(diag == 0)[0][0]
            return np.ravel_multi_index((ind, ind), (3, 3))
        elif abs(np.sum(diag)) == 2 and 0 in diag:
            ind = np.where(diag == 0)[0][0]
            action = np.ravel_multi_index((ind, ind), (3, 3))

        anti_diag = np.fliplr(board).diagonal()
        if np.sum(anti_diag) == 2 and 0 in anti_diag:
            ind = np.where(anti_diag == 0)[0][0]
            return np.ravel_multi_index((ind, 2 - ind), (3, 3))
        elif abs(np.sum(anti_diag)) == 2 and 0 in anti_diag:
            ind = np.where(anti_diag == 0)[0][0]
            action = np.ravel_multi_index((ind, 2 - ind), (3, 3))

        return action
