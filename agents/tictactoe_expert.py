import numpy as np


import torch
from torch.distributions import Categorical
from modules.models.inference_output import InferenceOutput

class TicTacToeBestAgent:
    def __init__(self, num_actions: int = 9, name="tictactoe_expert"):
        self.num_actions = num_actions
        self.name = name

    def obs_inference(self, obs: torch.Tensor, **kwargs) -> InferenceOutput:
        """
        Calculates the best move for TicTacToe and returns an InferenceOutput.
        """
        # Ensure obs is [B, C, H, W]
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        # Use info from kwargs if available for legal_moves
        info = kwargs.get("info", {})
        
        # We'll calculate the action for each element in the batch
        # Usually Evaluator uses B=1 for TicTacToe experts
        best_actions = []
        for b in range(batch_size):
            board = obs[b, 0] - obs[b, 1]
            
            # Default to random legal move if no obvious winning/blocking move
            # Support both batched (list of lists) and unbatched legal moves
            # Also handle boolean mask tensors
            legal_moves = info.get("legal_moves")
            if legal_moves is None:
                legal_moves = info.get("legal_moves_mask")
            
            # Extract legal moves for the current batch index 'b'
            if isinstance(legal_moves, torch.Tensor):
                # If it's a mask [B, A]
                if legal_moves.ndim == 2:
                    current_legal = torch.where(legal_moves[b])[0].cpu().numpy().tolist()
                else:
                    current_legal = torch.where(legal_moves)[0].cpu().numpy().tolist()
            elif isinstance(legal_moves, list):
                if len(legal_moves) > 0 and isinstance(legal_moves[0], (list, np.ndarray, torch.Tensor)):
                    # Batched list of lists
                    current_legal = legal_moves[b]
                    if isinstance(current_legal, torch.Tensor):
                        current_legal = current_legal.cpu().numpy().tolist()
                else:
                    current_legal = legal_moves
            else:
                current_legal = []

            # If legal_moves is missing, we must infer it from the board
            if not current_legal:
                current_legal = np.where(board.cpu().numpy().flatten() == 0)[0].tolist()
            
            action = np.random.choice(current_legal) if current_legal else 0
            
            # Horizontal and vertical checks
            found_move = False
            for i in range(3):
                # Row
                if not found_move and np.sum(board[i, :].cpu().numpy()) == 2 and 0 in board[i, :].cpu().numpy():
                    ind = np.where(board[i, :].cpu().numpy() == 0)[0][0]
                    action = np.ravel_multi_index((i, ind), (3, 3))
                    found_move = True
                elif not found_move and abs(np.sum(board[i, :].cpu().numpy())) == 2 and 0 in board[i, :].cpu().numpy():
                    ind = np.where(board[i, :].cpu().numpy() == 0)[0][0]
                    action = np.ravel_multi_index((i, ind), (3, 3))
                    # Don't set found_move to True here, because a winning move is better than a blocking move

                # Column
                if not found_move and np.sum(board[:, i].cpu().numpy()) == 2 and 0 in board[:, i].cpu().numpy():
                    ind = np.where(board[:, i].cpu().numpy() == 0)[0][0]
                    action = np.ravel_multi_index((ind, i), (3, 3))
                    found_move = True
                elif not found_move and abs(np.sum(board[:, i].cpu().numpy())) == 2 and 0 in board[:, i].cpu().numpy():
                    ind = np.where(board[:, i].cpu().numpy() == 0)[0][0]
                    action = np.ravel_multi_index((ind, i), (3, 3))

            # Diagonals
            if not found_move:
                diag = board.diagonal().cpu().numpy()
                if np.sum(diag) == 2 and 0 in diag:
                    ind = np.where(diag == 0)[0][0]
                    action = np.ravel_multi_index((ind, ind), (3, 3))
                    found_move = True
                elif abs(np.sum(diag)) == 2 and 0 in diag:
                    ind = np.where(diag == 0)[0][0]
                    action = np.ravel_multi_index((ind, ind), (3, 3))

            if not found_move:
                anti_diag = np.fliplr(board.cpu().numpy()).diagonal()
                if np.sum(anti_diag) == 2 and 0 in anti_diag:
                    ind = np.where(anti_diag == 0)[0][0]
                    action = np.ravel_multi_index((ind, 2 - ind), (3, 3))
                    found_move = True
                elif abs(np.sum(anti_diag)) == 2 and 0 in anti_diag:
                    ind = np.where(anti_diag == 0)[0][0]
                    action = np.ravel_multi_index((ind, 2 - ind), (3, 3))

            best_actions.append(action)

        # Create one-hot probabilities for the best actions
        probs = torch.zeros((batch_size, self.num_actions), device=obs.device)
        for b, a in enumerate(best_actions):
            probs[b, a] = 1.0
            
        return InferenceOutput(
            policy=Categorical(probs=probs),
            # Expert provides "ground truth" value if possible, here we'll just return 0.0
            value=torch.zeros((batch_size,), device=obs.device)
        )
        # Handle both (obs, info) tuple and raw obs for robustness
        obs = prediction[0] if isinstance(prediction, (tuple, list)) else prediction

        # Handle batch dimension if present (use first sample)
        if obs.ndim == 4:
            obs = obs[0]

        # Reconstruct board: +1 for current player, -1 for opponent, 0 otherwise
        # current player is Plane 0, opponent is Plane 1
        board = obs[0] - obs[1]
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
