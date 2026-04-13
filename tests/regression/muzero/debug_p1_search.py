import torch
import numpy as np
from envs.factories.tictactoe import tictactoe_factory
from registries import make_muzero_network, make_muzero_search_engine
from core import SingleBatchIterator
from search.backends.aos_search.tree import FlatTree
from search.backends.aos_search.batched_mcts import batched_mcts_step
from modules.world_models.inference_output import MuZeroNetworkState
import torch.distributions as dists

def debug_p1_search():
    print("=== Debugging MuZero P1 Search Perspective ===")
    DEVICE = torch.device("cpu")
    
    # 1. Setup Environment and Network (Matching Regression Test EXACTLY)
    env = tictactoe_factory()
    num_actions = env.action_space("player_1").n
    obs_dim = env.observation_space("player_1").shape
    
    # Exact hyperparams from test_muzero_tictactoe_regression.py
    ACTION_EMBEDDING_DIM = 32
    RESNET_FILTERS = [24, 24, 24]
    UNROLL_STEPS = 5
    
    agent_network = make_muzero_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        action_embedding_dim=ACTION_EMBEDDING_DIM,
        resnet_filters=RESNET_FILTERS,
        unroll_steps=UNROLL_STEPS,
        device=DEVICE,
    )
    agent_network.eval()
    
    # 2. Create a "P1 Must Block" State
    # P0 (X) at (1,1) and (1,2)
    # P1 (O) at (2,1)
    
    # Expected shape: (9, 3, 3)
    # Plane 0-1: Current frame (P0 pieces, P1 pieces)
    # Plane 2-3: Frame -1
    # Plane 4-5: Frame -2
    # Plane 6-7: Frame -3
    # Plane 8: Player Indicator (1.0 for P1)
    
    obs_raw = np.zeros((9, 3, 3), dtype=np.float32)
    obs_raw[0, 1, 1] = 1.0 # P0 pieces in current frame
    obs_raw[0, 1, 2] = 1.0 # P0 pieces in current frame
    obs_raw[1, 2, 1] = 1.0 # P1 pieces in current frame
    obs_raw[8, :, :] = 1.0 # Indicator for P1
    
    obs_t = torch.tensor(obs_raw).unsqueeze(0).to(DEVICE) # [1, 9, 3, 3]
    
    # 3. Model Inference at Root
    with torch.no_grad():
        outputs = agent_network.obs_inference(obs_t)
        
    print(f"Root Value Prediction: {outputs.value.item():.4f}")
    # Root to_play should be None from obs_inference for MuZero, 
    # but let's see what the ToPlay head says if we call it directly on the hidden state
    wm = agent_network.components["world_model"]
    tp_logits, _, _ = wm.to_play_head(outputs.network_state.dynamics)
    tp_pred = torch.argmax(tp_logits, dim=-1).item()
    print(f"Root Predicted ToPlay: {tp_pred} (Expected: 1 for P1)")

    # 4. Run MCTS from Root
    search_engine = make_muzero_search_engine(
        num_actions=num_actions,
        num_simulations=50,
        device=DEVICE,
    )
    
    print("\nRunning MCTS from P1 perspective (must block at (1,0))...")
    info = {"player": 1, "legal_moves": [0, 1, 2, 3, 6, 7, 8]} # (1,0) is index 3 in 0-8 flat mapping (row 1, col 0)
    
    # Inject into Search Engine
    root_v, exploratory_policy, target_policy, best_action, metadata = search_engine.run(
        obs_t, info, agent_network
    )
    
    print(f"MCTS Best Action: {best_action} (Index 3 corresponds to Row 1, Col 0)")
    print(f"MCTS Policy at root: {target_policy.tolist()}")
    
    # 5. Deep Inspection of Tree Signs
    # We'll run one simulation step manually to see the math
    tree = FlatTree.allocate(1, 512, num_actions, 0, DEVICE)
    # Dynamics hidden state needs to be cloned to escape inference mode if necessary
    h_state = outputs.network_state.dynamics.detach().clone()
    
    # Initialize the buffer with the expected structure ( dataclass/NamedTuple )
    # FlatTree.allocate usually returns an empty buffer if not careful, 
    # but here we manually assign it to match what AOS Search expects.
    tree.network_state_buffer = MuZeroNetworkState(
        dynamics=torch.zeros((1, 512, *h_state.shape[1:])),
        wm_memory=None
    )
    tree.network_state_buffer.dynamics[:, 0] = h_state
    tree.to_play[:, 0] = 1 # P1 at root
    tree.node_visits[:, 0] = 1
    tree.node_values[:, 0] = root_v
    
    # Run a few simulations and check the backpropped values
    with torch.no_grad():
        for _ in range(5):
            batched_mcts_step(
                tree, agent_network, max_depth=5, pb_c_init=1.25, pb_c_base=19652.0, 
                discount=0.99, search_batch_size=1, num_players=2
            )
    
    # Check tree.to_play for expanded nodes
    print("\nTree Inspection (First 10 nodes):")
    for i in range(10):
        if tree.node_visits[0, i] > 0:
            tp = tree.to_play[0, i].item()
            val = tree.node_values[0, i].item()
            visits = tree.node_visits[0, i].item()
            print(f"  Node {i}: Player={tp}, Value={val:.4f}, Visits={visits}")

if __name__ == "__main__":
    debug_p1_search()
