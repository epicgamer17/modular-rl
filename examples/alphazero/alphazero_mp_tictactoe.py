"""AlphaZero on PettingZoo TicTacToe (torch.multiprocessing Multiprocess Version)

================================================================================

Paper Reference:
    "Mastering Chess and Shogi by Self-Play with a General Reinforcement
    Learning Algorithm"
    Silver et al., Science 2018 (arXiv 2017: https://arxiv.org/abs/1712.01815)

Multiprocessing Architecture:
    - 3 Actor Processes: Each actor runs batched self-play across 5 parallel
    environments
      simultaneously using batched MCTS (root embeddings shape [5, 3, 3, 2]).
    - 1 Learner Process: Continuously samples minibatches from the
    SharedReplayBuffer,
      computes policy and value loss, steps the SGD optimizer, and updates
      shared weights.
    - 1 Async Evaluator Process: Periodically evaluates the shared model against
    both a Random baseline and a Perfect (negamax) opponent in the background,
    logging metrics to W&B asynchronously.
"""

import random
import time
from typing import Any, Callable, Dict, List, Tuple

from atomic_rl.action_selection import sample_distribution
from atomic_rl.buffers.replay import (
    circular_write_strategy_,
    init_buffer,
    uniform_sample,
)
from atomic_rl.envs.functions.tictactoe import (
    check_tictactoe_winner,
    embeddings_to_canonical,
    get_canonical_obs,
    tictactoe_dynamics_fn,
)
from atomic_rl.losses import cross_entropy_loss, mse_loss
from atomic_rl.networks import ResNetBackbone
from atomic_rl.search import get_mcts_visit_policy, mcts_search
from pettingzoo.classic import tictactoe_v3

from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb

# TODO: use get_legal_action_mask
# TODO: use argmax selector for evaluation
# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 20000
NUM_ACTORS = 3
ENVS_PER_ACTOR = 3  # 4 parallel vectorized environments per actor
MIN_BUFFER_SIZE = 64
EVAL_INTERVAL_STEPS = 250
PARAM_SYNC_INTERVAL = 100
NUM_MCTS_SIMULATIONS = 50

# MCTS PUCT Search Constants
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Temperature Schedule Constants
TEMP_THRESHOLD_MOVES = 5
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0

# Optimization & Network Parameters
BATCH_SIZE = 48
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_FILTERS = 24
NUM_RES_BLOCKS = 6

NUM_EVAL_GAMES = 100
SEED = 42


# ============================================================================
# 1. Dual-Head AlphaZero ResNet Neural Network
# ============================================================================
class TicTacToeNet(nn.Module):
    def __init__(
        self, num_filters: int = NUM_FILTERS, num_res_blocks: int = NUM_RES_BLOCKS
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)

        self.res_blocks = ResNetBackbone(
            in_channels=num_filters,
            num_filters=num_filters,
            num_blocks=num_res_blocks,
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 9),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.bn_in(self.conv_in(x)))
        features = self.res_blocks(features)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value.squeeze(-1)


# ============================================================================
# 2. Thread & Process-Safe Shared Replay Buffer
# ============================================================================


class SharedReplayBuffer:
    def __init__(self, capacity: int, shapes: Dict[str, Any], device: torch.device):
        self.buffer_state = init_buffer(capacity, shapes, device=device)
        self.buffer_state.data.share_memory_()

        self.lock = mp.Lock()
        self._pointer = mp.Value("i", 0)
        self._size = mp.Value("i", 0)

    def add_samples(self, samples_td: TensorDict):
        with self.lock:
            self.buffer_state.pointer = self._pointer.value
            self.buffer_state.size = self._size.value

            self.buffer_state, _ = circular_write_strategy_(
                self.buffer_state, samples_td
            )

            self._pointer.value = self.buffer_state.pointer
            self._size.value = self.buffer_state.size

    def sample_batch(
        self, rng_key: torch.Generator, batch_size: int, min_size: int = 0
    ) -> TensorDict:
        with self.lock:
            if self._size.value < min_size:
                return None

            self.buffer_state.size = self._size.value
            return uniform_sample(self.buffer_state, rng_key, batch_size)

    @property
    def size(self) -> int:
        return self._size.value


# ============================================================================
# 3. Multiprocessing Workers
# ============================================================================
# TODO: i think we have a function for this already. Use it.
def apply_action_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Masks illegal actions to the minimum finite value of the dtype."""
    min_logit = torch.finfo(logits.dtype).min
    return logits.masked_fill(~legal_mask, min_logit)


def create_recurrent_fn(local_model: nn.Module) -> Callable:
    """Creates an mctx-compliant recurrent_fn for PyTorch MCTS."""

    def recurrent_fn(
        actions_taken: torch.Tensor, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Step environment dynamics (AlphaZero simulator or MuZero world model)
        (
            next_embed,
            reward,
            next_to_play,
            is_terminal,
            next_legal_mask,
        ) = tictactoe_dynamics_fn(embeddings, actions_taken)

        # 2. Evaluate policy logits and value network predictions for the next state
        with torch.no_grad():
            canonical_next = embeddings_to_canonical(next_embed)
            logits, value = local_model(canonical_next)

        # 3. Mask illegal actions for the internal child node using min_logit
        masked_logits = apply_action_mask(logits, next_legal_mask)

        # 4. Handle discounts and terminal states
        # For alternating-turn zero-sum games: non-terminal discount is -1.0, terminal is 0.0
        discount = torch.where(
            is_terminal,
            torch.zeros_like(reward),
            -torch.ones_like(reward),
        )

        # 5. Zero out network value for terminal states (terminal nodes have no future return)
        value = torch.where(is_terminal, torch.zeros_like(value), value)

        return masked_logits, value, reward, discount, next_embed

    return recurrent_fn


def actor_worker(
    actor_id: int,
    num_actors: int,
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    envs_per_actor: int = ENVS_PER_ACTOR,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    # Set worker seed
    worker_seed = SEED + actor_id * 1000
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)

    boards = torch.zeros(envs_per_actor, 3, 3, device=device)
    players = torch.zeros(envs_per_actor, dtype=torch.long, device=device)
    move_counts = [0] * envs_per_actor
    trajectories = [[] for _ in range(envs_per_actor)]

    step_counter = 0
    recurrent_fn = create_recurrent_fn(local_model)

    while True:
        step_counter += 1
        if step_counter % PARAM_SYNC_INTERVAL == 0:
            local_model.load_state_dict(shared_model.state_dict())

        # TODO: I don't love this part. Ideally we use the pettingzoo obs and stuff directly.
        # Construct root embeddings for search batch size [B, 3, 3, 2]
        root_embed = boards.new_zeros(envs_per_actor, 3, 3, 2)
        root_embed[..., 0] = boards
        root_embed[..., 1] = players.view(-1, 1, 1).expand(-1, 3, 3).float()
        legal_mask = boards.view(envs_per_actor, -1) == 0.0

        # Root Model Prediction
        with torch.no_grad():
            canonical_x = embeddings_to_canonical(root_embed)
            root_logits, root_value = local_model(canonical_x)

        # Execute MCTS Search
        search_action, action_probs, tree = mcts_search(
            root_embeddings=root_embed,
            root_logits=root_logits,
            root_value=root_value,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            num_actions=9,
            legal_mask=legal_mask,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            dirichlet_alpha=DIRICHLET_ALPHA,
            pb_c_init=C_PUCT,
            temperature=1.0,  # Keeps target policy distributions smooth
        )

        completed_samples = []

        # Step each env in the search batch forward
        for b_idx in range(envs_per_actor):
            move_counts[b_idx] += 1
            curr_player = players[b_idx].item()

            # Target policy from search
            target_policy = action_probs[b_idx]

            # Select action according to move-count temperature schedule
            temp = (
                TEMPERATURE_EXPLORATION
                if move_counts[b_idx] <= TEMP_THRESHOLD_MOVES
                else TEMPERATURE_EXPLOITATION
            )

            if temp == 0.0:
                action_idx = target_policy.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs=target_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action_idx = action_idx_t.item()

            canonical_obs = get_canonical_obs(boards[b_idx], curr_player).squeeze(0)

            trajectories[b_idx].append(
                {
                    "state": canonical_obs.cpu(),
                    "target_policy": target_policy.cpu(),
                    "player": curr_player,
                }
            )

            # Fast internal dynamics step
            row, col = action_idx // 3, action_idx % 3
            piece = 1.0 if curr_player == 0 else -1.0
            boards[b_idx, row, col] = piece

            winner, is_term = check_tictactoe_winner(boards[b_idx].unsqueeze(0))

            if is_term.item():
                p0_reward = winner.item()
                p1_reward = -p0_reward

                for step in trajectories[b_idx]:
                    pl = step["player"]
                    z = p0_reward if pl == 0 else p1_reward
                    completed_samples.append(
                        {
                            "state": step["state"],
                            "target_policy": step["target_policy"],
                            "target_value": torch.tensor([z], dtype=torch.float32),
                        }
                    )

                # Reset environment for new game
                boards[b_idx].zero_()
                players[b_idx] = 0
                move_counts[b_idx] = 0
                trajectories[b_idx] = []
            else:
                players[b_idx] = 1 - curr_player

        if len(completed_samples) > 0:
            batch_td = TensorDict(
                {
                    "state": torch.stack([s["state"] for s in completed_samples]),
                    "target_policy": torch.stack(
                        [s["target_policy"] for s in completed_samples]
                    ),
                    "target_value": torch.stack(
                        [s["target_value"] for s in completed_samples]
                    ),
                },
                batch_size=[len(completed_samples)],
            ).to(device)
            buffer.add_samples(batch_td)


def evaluator_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    num_eval_games: int = NUM_EVAL_GAMES,
    eval_interval_steps: int = EVAL_INTERVAL_STEPS,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    recurrent_fn = create_recurrent_fn(local_model)

    while True:
        time.sleep(2.0)
        local_model.load_state_dict(shared_model.state_dict())

        # Evaluate vs Random
        p1_wins, p1_draws, p1_losses = 0, 0, 0
        p2_wins, p2_draws, p2_losses = 0, 0, 0

        for game_i in range(num_eval_games):
            az_player = 0 if game_i % 2 == 0 else 1
            az_agent_name = "player_1" if az_player == 0 else "player_2"
            env = tictactoe_v3.env()
            env.reset()

            board_3x3 = torch.zeros(3, 3, device=device)
            az_reward = 0.0

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                if agent == az_agent_name and reward != 0:
                    az_reward = reward

                if termination or truncation:
                    env.step(None)
                    continue

                # TODO: this is how the action mask logic should be done in the self play loop too. Why is it not done like this?
                player = 0 if agent == "player_1" else 1
                action_mask = torch.tensor(
                    obs["action_mask"], device=device, dtype=torch.bool
                )
                legal_actions = action_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

                if player == az_player:
                    root_embed = board_3x3.new_zeros(1, 3, 3, 2)
                    root_embed[0, ..., 0] = board_3x3
                    root_embed[0, ..., 1] = float(player)

                    with torch.no_grad():
                        canonical_x = embeddings_to_canonical(root_embed)
                        root_logits, root_value = local_model(canonical_x)

                    search_action, action_probs, tree = mcts_search(
                        root_embeddings=root_embed,
                        root_logits=root_logits,
                        root_value=root_value,
                        recurrent_fn=recurrent_fn,
                        num_simulations=NUM_MCTS_SIMULATIONS,
                        num_actions=9,
                        legal_mask=action_mask.unsqueeze(0),
                        dirichlet_epsilon=0.0,
                        pb_c_init=C_PUCT,
                        temperature=0.0,
                    )

                    action_idx = search_action[0].item()
                else:
                    action_idx = random.choice(legal_actions)

                row, col = action_idx // 3, action_idx % 3
                piece = 1.0 if player == 0 else -1.0
                board_3x3[row, col] = piece
                env.step(action_idx)

            if az_player == 0:
                if az_reward > 0:
                    p1_wins += 1
                elif az_reward < 0:
                    p1_losses += 1
                else:
                    p1_draws += 1
            else:
                if az_reward > 0:
                    p2_wins += 1
                elif az_reward < 0:
                    p2_losses += 1
                else:
                    p2_draws += 1

        total_p1 = max(1, p1_wins + p1_draws + p1_losses)
        total_p2 = max(1, p2_wins + p2_draws + p2_losses)
        total_all = num_eval_games

        eval_metrics = {
            "eval/win_rate": (p1_wins + p2_wins) / total_all,
            "eval/draw_rate": (p1_draws + p2_draws) / total_all,
            "eval/loss_rate": (p1_losses + p2_losses) / total_all,
            "eval/p1_win_rate": p1_wins / total_p1,
            "eval/p1_draw_rate": p1_draws / total_p1,
            "eval/p1_loss_rate": p1_losses / total_p1,
            "eval/p2_win_rate": p2_wins / total_p2,
            "eval/p2_draw_rate": p2_draws / total_p2,
            "eval/p2_loss_rate": p2_losses / total_p2,
        }
        eval_queue.put(eval_metrics)


def learner_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    eval_queue: mp.Queue,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.train()
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = torch.optim.Adam(
        local_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    wandb.init(
        project="alphazero-tictactoe",
        name=f"alphazero_mp_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}_actors{NUM_ACTORS}",
        config={
            "total_training_steps": max_steps,
            "num_actors": NUM_ACTORS,
            "envs_per_actor": ENVS_PER_ACTOR,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "c_puCT": C_PUCT,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": SEED,
        },
    )

    step_count = 0
    latest_eval_metrics = {}

    while step_count < max_steps:
        minibatch = buffer.sample_batch(rng_key, batch_size, MIN_BUFFER_SIZE)
        if minibatch is None:
            time.sleep(0.05)
            continue

        states = minibatch["state"].to(device)
        target_policies = minibatch["target_policy"].to(device)
        target_values = minibatch["target_value"].to(device)

        policy_logits, predicted_value = local_model(states)

        raw_p_loss, _ = cross_entropy_loss(policy_logits, target_policies)
        policy_loss = raw_p_loss.mean()

        raw_v_loss, _ = mse_loss(predicted_value.view(-1), target_values.view(-1))
        value_loss = raw_v_loss.mean()

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update shared model memory
        shared_model.load_state_dict(local_model.state_dict())
        step_count += 1

        # Check for fresh evaluation metrics from background evaluator
        while not eval_queue.empty():
            try:
                latest_eval_metrics = eval_queue.get_nowait()
            except Exception:
                break

        log_dict = {
            "global_step": step_count,
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "buffer/size": buffer.size,
        }
        log_dict.update(latest_eval_metrics)

        if step_count % 100 == 0:
            print(
                f"[Learner Step {step_count}/{max_steps}] Loss: {loss.item():.4f} | Buffer: {buffer.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


# ============================================================================
# 4. Main Multiprocessing Setup
# ============================================================================


def model_creator_fn() -> TicTacToeNet:
    return TicTacToeNet(num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS)


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device_str = "cpu"
    device = torch.device(device_str)

    shared_model = model_creator_fn().to(device)
    shared_model.share_memory()

    buffer_shapes = {
        "state": (3, 3, 3),
        "target_policy": (9,),
        "target_value": (1,),
    }
    buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device)
    eval_queue = mp.Queue()

    processes = []

    # Start Learner Process
    learner_p = mp.Process(
        target=learner_worker,
        args=(model_creator_fn, shared_model, buffer, eval_queue),
        kwargs={"device_str": device_str},
    )
    learner_p.start()
    processes.append(learner_p)

    # Start 3 Actor Processes
    for actor_id in range(NUM_ACTORS):
        actor_p = mp.Process(
            target=actor_worker,
            args=(actor_id, NUM_ACTORS, model_creator_fn, shared_model, buffer),
            kwargs={"device_str": device_str},
        )
        actor_p.start()
        processes.append(actor_p)

    # Start Async Evaluator Process
    eval_p = mp.Process(
        target=evaluator_worker,
        args=(model_creator_fn, shared_model, eval_queue),
        kwargs={"device_str": device_str},
    )
    eval_p.daemon = True
    eval_p.start()
    processes.append(eval_p)

    print(
        f"Launched AlphaZero Multiprocessing Pipeline ({NUM_ACTORS} Actors, 1 Learner, 1 Evaluator)."
    )
    learner_p.join()

    # Clean termination of actors
    for p in processes:
        if p.is_alive():
            p.terminate()

    print("AlphaZero Multiprocessing Training Completed Successfully.")


if __name__ == "__main__":
    main()
