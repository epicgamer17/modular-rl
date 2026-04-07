import numpy as np
from utils.utils import action_mask_to_legal_moves

def action_mask_to_info(state, info, current_player):
    info["legal_moves"] = action_mask_to_legal_moves(state["action_mask"])
    info["action_mask"] = state["action_mask"]
    info["player"] = current_player
    if "observation" in state:
        state = state["observation"]
    return state
