from tabnanny import check
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy


class CheckersEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.window_size = (512, 512)  # The size of the PyGame window

        # Observations are planes.
        # The first plane represents player 1s tokens, the second player 2s and the third encodes the current players turn.
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                5,
                8,
                8,
            ),  # 5 planes: player 1, player 2, player 1 king, player 2 king, current player
            dtype=np.float64,
        )
        print(self.observation_space)

        # We have 9 actions, corresponding to each cell
        self.action_space = spaces.Discrete(8 * 32 + 1)
        # 0, 1, 2, 3, 4, 5, 6, 7 * 32 squares + 1
        # from, action_type = num // 8 == from_square?, num % 8 == action type
        # 0 1 0 1 0 1 0 1
        # 1 0 1 0 1 0 1 0
        # 0 1 0 1 0 1 0 1
        # 0 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0 0
        # 2 0 2 0 2 0 2 0
        # 0 2 0 2 0 2 0 2
        # 2 0 2 0 2 0 2 0

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
        return copy.deepcopy(self._grid)

    def _get_info(self):
        return {
            "legal_moves": self._legal_moves,
            "player": self._current_player,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set a blank board
        self._grid = np.zeros(
            (
                5,
                8,
                8,
            )
        )
        # set player ones and twos tokens
        for i in range(0, 8, 2):
            self._grid[1, 1, i] = 1
            self._grid[1, 5, i] = 1
            self._grid[0, 7, i] = 1
        for i in range(1, 8, 2):
            self._grid[1, 0, i] = 1
            self._grid[0, 2, i] = 1
            self._grid[0, 6, i] = 1
        self._grid[4, :, :] = 0  # It's player 1's turn
        self._current_player = 0
        # Reset legal moves
        # 0 1 0 1 0 1 0 1
        # 1 0 1 0 1 0 1 0
        # 0 1 0 1 0 1 0 1
        # 0 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0 0
        # 2 0 2 0 2 0 2 0
        # 0 2 0 2 0 2 0 2
        # 2 0 2 0 2 0 2 0
        # 0, 1, 2, 3, 4, 5, 6, 7 * 32 squares + 1

        self._legal_moves = np.array(
            [
                20 * 8 + 1,
                21 * 8 + 0,
                21 * 8 + 1,
                22 * 8 + 0,
                22 * 8 + 1,
                23 * 8 + 0,
                23 * 8 + 1,
            ]
        )
        # self._legal_moves_p2 = np.array(
        #     [
        #         8 * 8 + 2,
        #         8 * 8 + 3,
        #         9 * 8 + 2,
        #         9 * 8 + 3,
        #         10 * 8 + 2,
        #         10 * 8 + 3,
        #         11 * 8 + 3,
        #     ]
        # )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if action < 0 or action > 8:
            raise ValueError("Action must be between 0 and 8")
        if action not in self._legal_moves:
            # could return a negative reward
            raise ValueError(
                "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
            )

        # convert action to from_square and action_type
        from_square = action // 8
        action_type = action % 8
        # convert from square to row and column
        # from_square 0-3 is row 0, from_square 4-7 is row 1, etc.
        # from_square 0 is col 1, from_square 1 is col 3, from_square 2 is col 5, from_square 3 is col 7, from_square 4 is col 0, etc.
        from_row = from_square // 4
        from_col = (from_square % 4) * 2 + (1 - from_row % 2)
        # move token
        capture = False
        if action_type == 0:
            to_row = from_row - 1
            to_col = from_col - 1
        elif action_type == 1:
            to_row = from_row - 1
            to_col = from_col + 1
        elif action_type == 2:
            to_row = from_row + 1
            to_col = from_col + 1
        elif action_type == 3:
            to_row = from_row + 1
            to_col = from_col - 1
        elif action_type == 4:
            to_row = from_row - 2
            to_col = from_col - 2
            capture = True
        elif action_type == 5:
            to_row = from_row - 2
            to_col = from_col + 2
            capture = True
        elif action_type == 6:
            to_row = from_row + 2
            to_col = from_col + 2
            capture = True
        elif action_type == 7:
            to_row = from_row + 2
            to_col = from_col - 2
            capture = True
        # check if move is valid?

        # update board state
        self._grid[0, to_row, to_col] = self._grid[0, from_row, from_col]
        self._grid[0, from_row, from_col] = 0
        self._grid[1, to_row, to_col] = self._grid[1, from_row, from_col]
        self._grid[1, from_row, from_col] = 0
        if to_row == 0 and self._grid[0, to_row, to_col] == 1:
            self._grid[1, to_row, to_col] = 1
            self._grid[0, to_row, to_col] = 0

        if capture:
            self._grid[2, (from_row + to_row) // 2, (from_col + to_col) // 2] = 0
            self._grid[3, (from_row + to_row) // 2, (from_col + to_col) // 2] = 0

        # output next player's token first (since that's the one we're inputting to)
        current_player_tokens = copy.deepcopy(self._grid[0, :, :])
        self._grid[0, :, :] = self._grid[:, :, 2]
        self._grid[2, :, :] = current_player_tokens

        current_player_kings = copy.deepcopy(self._grid[1, :, :])
        self._grid[1, :, :] = self._grid[:, :, 3]
        self._grid[3, :, :] = current_player_kings

        # flip board vertically
        self._grid[0, :, :] = np.flip(self._grid[0, :, :], axis=0)
        self._grid[1, :, :] = np.flip(self._grid[1, :, :], axis=0)
        self._grid[2, :, :] = np.flip(self._grid[2, :, :], axis=0)
        self._grid[3, :, :] = np.flip(self._grid[3, :, :], axis=0)

        # encode turn as 1 or 0
        self._grid[4, :, :] = 1 - self._grid[4, :, :]
        self._current_player = (self._current_player + 1) % 2
        # An episode is done iff there is a winner or the board is full

        # output new legal moves
        self._legal_moves = []
        for i in range(32):
            check_row = i // 4
            check_col = (i % 4) * 2 + (1 - check_row % 2)
            if self._grid[0, check_row, check_col] == 1:
                if (
                    check_row - 1 >= 0
                    and check_col - 1 >= 0
                    and (
                        self._grid[0, check_row - 1, check_col - 1] == 0
                        and self._grid[1, check_row - 1, check_col - 1] == 0
                        and self._grid[2, check_row - 1, check_col - 1] == 0
                        and self._grid[3, check_row - 1, check_col - 1] == 0
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8)
                if (
                    check_row - 1 >= 0
                    and check_col + 1 < 8
                    and (
                        self._grid[0, check_row - 1, check_col + 1] == 0
                        and self._grid[1, check_row - 1, check_col + 1] == 0
                        and self._grid[2, check_row - 1, check_col + 1] == 0
                        and self._grid[3, check_row - 1, check_col + 1] == 0
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 1)
                if (
                    check_row - 2 >= 0
                    and check_col - 2 >= 0
                    and (
                        self._grid[0, check_row - 2, check_col - 2] == 0
                        and self._grid[1, check_row - 2, check_col - 2] == 0
                        and self._grid[2, check_row - 2, check_col - 2] == 0
                        and self._grid[3, check_row - 2, check_col - 2] == 0
                        and (
                            self._grid[2, check_row - 1, check_col - 1] == 1
                            or self._grid[3, check_row - 1, check_col - 1] == 1
                        )
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 4)
                if (
                    check_row - 2 >= 0
                    and check_col + 2 < 8
                    and (
                        self._grid[0, check_row - 2, check_col + 2] == 0
                        and self._grid[1, check_row - 2, check_col + 2] == 0
                        and self._grid[2, check_row - 2, check_col + 2] == 0
                        and self._grid[3, check_row - 2, check_col + 2] == 0
                        and (
                            self._grid[2, check_row - 1, check_col + 1] == 1
                            or self._grid[3, check_row - 1, check_col + 1] == 1
                        )
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 5)

            if self._grid[1, check_row, check_col] == 1:
                if (
                    check_row - 1 >= 0
                    and check_col - 1 >= 0
                    and (
                        self._grid[0, check_row - 1, check_col - 1] == 0
                        and self._grid[1, check_row - 1, check_col - 1] == 0
                        and self._grid[2, check_row - 1, check_col - 1] == 0
                        and self._grid[3, check_row - 1, check_col - 1] == 0
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 5)
                if (
                    check_row - 1 >= 0
                    and check_col + 1 < 8
                    and (
                        self._grid[0, check_row - 1, check_col + 1] == 0
                        and self._grid[1, check_row - 1, check_col + 1] == 0
                        and self._grid[2, check_row - 1, check_col + 1] == 0
                        and self._grid[3, check_row - 1, check_col + 1] == 0
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 4)
                if (
                    check_row + 1 < 8
                    and check_col + 1 < 8
                    and (
                        self._grid[0, check_row + 1, check_col + 1] == 0
                        and self._grid[1, check_row + 1, check_col + 1] == 0
                        and self._grid[2, check_row + 1, check_col + 1] == 0
                        and self._grid[3, check_row + 1, check_col + 1] == 0
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 2)
                if (
                    check_row + 1 < 8
                    and check_col - 1 >= 0
                    and (
                        self._grid[0, check_row + 1, check_col - 1] == 0
                        and self._grid[1, check_row + 1, check_col - 1] == 0
                        and self._grid[2, check_row + 1, check_col - 1] == 0
                        and self._grid[3, check_row + 1, check_col - 1] == 0
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 3)
                if (
                    check_row - 2 >= 0
                    and check_col - 2 >= 0
                    and (
                        self._grid[0, check_row - 2, check_col - 2] == 0
                        and self._grid[1, check_row - 2, check_col - 2] == 0
                        and self._grid[2, check_row - 2, check_col - 2] == 0
                        and self._grid[3, check_row - 2, check_col - 2] == 0
                        and (
                            self._grid[2, check_row - 1, check_col - 1] == 1
                            or self._grid[3, check_row - 1, check_col - 1] == 1
                        )
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 6)
                if (
                    check_row - 2 >= 0
                    and check_col + 2 < 8
                    and (
                        self._grid[0, check_row - 2, check_col + 2] == 0
                        and self._grid[1, check_row - 2, check_col + 2] == 0
                        and self._grid[2, check_row - 2, check_col + 2] == 0
                        and self._grid[3, check_row - 2, check_col + 2] == 0
                        and (
                            self._grid[2, check_row - 1, check_col + 1] == 1
                            or self._grid[3, check_row - 1, check_col + 1] == 1
                        )
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 7)
                if (
                    check_row + 2 < 8
                    and check_col + 2 < 8
                    and (
                        self._grid[0, check_row + 2, check_col + 2] == 0
                        and self._grid[1, check_row + 2, check_col + 2] == 0
                        and self._grid[2, check_row + 2, check_col + 2] == 0
                        and self._grid[3, check_row + 2, check_col + 2] == 0
                        and (
                            self._grid[2, check_row + 1, check_col + 1] == 1
                            or self._grid[3, check_row + 1, check_col + 1] == 1
                        )
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 2)
                if (
                    check_row + 2 < 8
                    and check_col - 2 >= 0
                    and (
                        self._grid[0, check_row + 2, check_col - 2] == 0
                        and self._grid[1, check_row + 2, check_col - 2] == 0
                        and self._grid[2, check_row + 2, check_col - 2] == 0
                        and self._grid[3, check_row + 2, check_col - 2] == 0
                        and (
                            self._grid[2, check_row + 1, check_col - 1] == 1
                            or self._grid[3, check_row + 1, check_col - 1] == 1
                        )
                    )
                ):
                    self._legal_moves = np.append(self._legal_moves, i * 8 + 3)
        # No legal moves means game over in checkers, not a time-limit truncation.
        has_winner = self.winner()
        no_legal_moves = len(self._legal_moves) == 0
        terminated = has_winner or no_legal_moves
        truncated = False
        if terminated:
            if self._grid[2, 0, 0] == 0:
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
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size[0] / 8
        )  # The size of a single grid square in pixels

        # First we draw the X's and 0's
        turn = int(self._grid[4, 0, 0])
        for i in range(8):
            for j in range(8):
                if self._grid[2 * turn, i, j] == 1:
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
                if self._grid[2 * turn + 1, i, j] == 1:
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.165) * pix_square_size, (i + 0.165) * pix_square_size),
                        ((j + 0.165) * pix_square_size, (i + 0.835) * pix_square_size),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.165) * pix_square_size, (i + 0.835) * pix_square_size),
                        ((j + 0.835) * pix_square_size, (i + 0.835) * pix_square_size),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.835) * pix_square_size, (i + 0.835) * pix_square_size),
                        ((j + 0.835) * pix_square_size, (i + 0.165) * pix_square_size),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.835) * pix_square_size, (i + 0.165) * pix_square_size),
                        ((j + 0.165) * pix_square_size, (i + 0.165) * pix_square_size),
                    )

                if self._grid[2 - 2 * turn, i, j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 0),
                        (
                            j * pix_square_size + pix_square_size // 2,
                            i * pix_square_size + pix_square_size // 2,
                        ),
                        pix_square_size // 3,
                        width=3,
                    )
                if self._grid[3 - 2 * turn, i, j] == 1:
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.165) * pix_square_size, (i + 0.165) * pix_square_size),
                        ((j + 0.165) * pix_square_size, (i + 0.835) * pix_square_size),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.165) * pix_square_size, (i + 0.835) * pix_square_size),
                        ((j + 0.835) * pix_square_size, (i + 0.835) * pix_square_size),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.835) * pix_square_size, (i + 0.835) * pix_square_size),
                        ((j + 0.835) * pix_square_size, (i + 0.165) * pix_square_size),
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((j + 0.835) * pix_square_size, (i + 0.165) * pix_square_size),
                        ((j + 0.165) * pix_square_size, (i + 0.165) * pix_square_size),
                    )
        # Finally, add some gridlines
        for i in range(8):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, i * pix_square_size),
                (self.window_size[0], i * pix_square_size),
            )
        for j in range(8):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (j * pix_square_size, 0),
                (j * pix_square_size, self.window_size[1]),
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
        if np.sum(self._grid[0, :, :]) == 0:
            return True
        if np.sum(self._grid[1, :, :]) == 0:
            return True
        if np.sum(self._grid[2, :, :]) == 0:
            return True
        if np.sum(self._grid[3, :, :]) == 0:
            return True
        return False
