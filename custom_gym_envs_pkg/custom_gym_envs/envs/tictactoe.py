import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=3, win_length=3, encode_player_turn=True):
        self.size = size  # The size of the square grid
        self.win_length = win_length  # The number of consecutive tokens needed to win
        self.window_size = 512  # The size of the PyGame window

        # Observations are planes.
        # The first plane represents player 1s tokens, the second player 2s and the third encodes the current players turn. (if encode_player_turn == True)
        self.encode_player_turn = encode_player_turn
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3 if encode_player_turn else 2, self.size, self.size),
            dtype=np.uint8,
        )

        # We have 9 actions, corresponding to each cell
        self.action_space = spaces.Discrete(self.size * self.size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.encode_player_turn:
            return copy.deepcopy(self._grid)
        else:
            return copy.deepcopy(self._grid[:2, :, :])

    def _get_info(self):
        return {
            "legal_moves": self._legal_moves,
            "player": self._current_player,
            "step": self._step_count,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set a blank board
        self._grid = np.zeros((3, self.size, self.size))
        self._grid[2, :, :] = 0  # It's player 1's turn
        self._current_player = 0

        self._step_count = 0

        # Reset legal moves
        self._legal_moves = np.array(list(range(self.action_space.n)))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        illegal_move = False
        if action < 0 or action > self.size * self.size - 1:
            raise ValueError("Action must be between 0 and 8")
            illegal_move = True
        if action not in self._legal_moves:
            # could return a negative reward
            # raise ValueError(
            #     "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
            # )
            print("Illegal move {} Legal Moves {}".format(action, self._legal_moves))
            illegal_move = True
        if illegal_move:
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                self._render_frame()

            reward = [0, 0]
            reward[self._current_player] = -1

            return observation, reward, False, False, info
        self._step_count += 1
        # output next player's token first (since that's the one we're inputting to)
        current_player_board = copy.deepcopy(self._grid[0, :, :])
        self._grid[0, :, :] = self._grid[1, :, :]
        self._grid[1, :, :] = current_player_board
        # print(self._grid[:, :, 0])
        # print(self._grid[:, :, 1])
        # print(self._grid[:, :, 2])
        # print("================")
        self._grid[1, action // self.size, action % self.size] = 1

        self._legal_moves = self._legal_moves[self._legal_moves != action]
        # encode turn as 1 or 0
        self._current_player = (self._current_player + 1) % 2
        self._grid[2, :, :] = 1 - self._grid[2, :, :]
        # Draw by full board is a true terminal game outcome, not a truncation.
        has_winner = self.winner()
        is_draw = len(self._legal_moves) == 0 and not has_winner
        terminated = has_winner or is_draw
        truncated = False
        if terminated:
            if self._current_player == 0:
                reward = [-1, 1]
            else:
                reward = [1, -1]
        else:
            reward = [0, 0]
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the X's and 0's
        turn = int(self._grid[2, 0, 0])
        for i in range(self.size):
            for j in range(self.size):
                if self._grid[turn, i, j] == 1:
                    pygame.draw.line(
                        canvas,
                        (0, 0, 255),
                        ((j + 0.165) * pix_square_size, (i + 0.165) * pix_square_size),
                        ((j + 0.835) * pix_square_size, (i + 0.835) * pix_square_size),
                        width=3,
                    )
                    pygame.draw.line(
                        canvas,
                        (0, 0, 255),
                        ((j + 0.835) * pix_square_size, (i + 0.165) * pix_square_size),
                        ((j + 0.165) * pix_square_size, (i + 0.835) * pix_square_size),
                        width=3,
                    )
                if self._grid[1 - turn, i, j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        (
                            j * pix_square_size + pix_square_size // 2,
                            i * pix_square_size + pix_square_size // 2,
                        ),
                        pix_square_size // 3,
                        width=3,
                    )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def winner(self):
        for i in range(self.size):
            for j in range(self.size):
                if self._grid[1, i, j] == 1:
                    if self._check_win(i, j):
                        return True
        return False

    def _check_win(self, i, j):
        # Check row
        if np.all([self._grid[1, i, k] for k in range(self.win_length)]):
            return True
        # Check column
        if np.all([self._grid[1, k, j] for k in range(self.win_length)]):
            return True

        # if np.all(self._grid[i, :, 1] == 1):
        #     return True
        # # Check column
        # if np.all(self._grid[:, j, 1] == 1):
        #     return True

        # Check diagonals for any cell
        if i + self.win_length <= self.size and j + self.win_length <= self.size:
            if np.all([self._grid[1, i + k, j + k] for k in range(self.win_length)]):
                return True
        if i - self.win_length >= -1 and j + self.win_length <= self.size:
            if np.all([self._grid[1, i - k, j + k] for k in range(self.win_length)]):
                return True
        if i + self.win_length <= self.size and j - self.win_length >= -1:
            if np.all([self._grid[1, i + k, j - k] for k in range(self.win_length)]):
                return True
        if i - self.win_length >= -1 and j - self.win_length >= -1:
            if np.all([self._grid[1, i - k, j - k] for k in range(self.win_length)]):
                return True
        return False
