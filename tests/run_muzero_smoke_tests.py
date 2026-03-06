import torch
from tests.trainers.test_trainer_muzero_end_to_end_smoke import (
    test_muzero_cartpole_smoke,
    test_muzero_tictactoe_smoke,
)

if __name__ == "__main__":
    print("Running MuZero CartPole smoke test...")
    test_muzero_cartpole_smoke()
    print("CartPole smoke test passed!")

    print("Running MuZero TicTacToe smoke test...")
    test_muzero_tictactoe_smoke()
    print("TicTacToe smoke test passed!")

    print("All MuZero smoke tests passed!")
