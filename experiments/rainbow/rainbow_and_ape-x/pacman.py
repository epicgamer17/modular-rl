import gymnasium as gym
import sys
import random
from collections import defaultdict
import copy
import math
from operator import itemgetter
import os
import matplotlib

from torch.optim.sgd import SGD
from torch.optim.adam import Adam

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy
import pickle

from typing import Iterable, Tuple
from datetime import datetime

import torch
from torch import nn, Tensor

import numpy as np
import numpy.typing as npt

import itertools
from hyperopt import space_eval

import pandas as pd


# from ....replay_buffers.base_replay_buffer import Game
# from replay_buffers.segment_tree import SumSegmentTree


def normalize_policies(policies: torch.float32):
    # print(policies)
    policy_sums = policies.sum(axis=-1, keepdims=True)
    # print(policy_sums)
    policies = policies / policy_sums
    return policies


def action_mask(
    actions: Tensor, legal_moves, mask_value: float = 0, device="cpu"
) -> Tensor:
    """
    Mask actions that are not legal moves
    actions: Tensor, probabilities of actions or q-values
    """
    assert isinstance(
        legal_moves, list
    ), "Legal moves should be a list got {} of type {}".format(
        legal_moves, type(legal_moves)
    )

    # add a dimension if the legal moves are not a list of lists
    # if len(legal_moves) != actions.shape[0]:
    #     legal_moves = [legal_moves]
    assert (
        len(legal_moves) == actions.shape[0]
    ), "Legal moves should be the same length as the batch size"

    mask = torch.zeros_like(actions, dtype=torch.bool).to(device)
    for i, legal in enumerate(legal_moves):
        mask[i, legal] = True
    # print(mask)
    # print(actions)
    # actions[mask == 0] = mask_value
    actions = torch.where(mask, actions, torch.tensor(mask_value).to(device)).to(device)
    # print(mask)
    return actions


def clip_low_prob_actions(actions: Tensor, low_prob: float = 0.01) -> Tensor:
    """
    Clip actions with probability lower than low_prob to 0
    actions: Tensor, probabilities of actions
    """
    # print("Actions in low prob func", actions)
    if low_prob == 0:
        return actions
    mask = actions < low_prob
    # print("Mask", mask)
    actions = torch.where(mask, 0.0, actions)
    # print("Actions after clipping", actions)
    return actions


def get_legal_moves(info: dict | list[dict]):
    # print(info)
    if isinstance(info, dict):
        return [info["legal_moves"] if "legal_moves" in info else None]
    else:
        return [(i["legal_moves"] if "legal_moves" in i else None) for i in info]


def normalize_images(image: Tensor) -> Tensor:
    """Preprocessing step to normalize image with 8-bit (0-255) color inplace.
    Modifys the original tensor

    Args:
        image (Tensor): An 8-bit color image

    Returns:
        Tensor: The tensor divided by 255
    """
    # Return a copy of the tensor divided by 255
    normalized_image = image.div_(255)
    return normalized_image


def make_stack(item: Tensor) -> Tensor:
    """Convert a tensor of shape (*) to (1, *). Does not copy the data; instead,
    returns a view of the original tensor.

    Args:
        item (Tensor):

    Returns:
        Tensor: A view of the original tensor.
    """
    #
    return item.view(1, *item.shape)


def update_per_beta(
    per_beta: float, per_beta_final: float, per_beta_steps: int, initial_per_beta: int
):
    # could also use an initial per_beta instead of current (multiply below equation by current step)
    if per_beta < per_beta_final:
        clamp_func = min
    else:
        clamp_func = max
    per_beta = clamp_func(
        per_beta_final,
        per_beta + (per_beta_final - initial_per_beta) / (per_beta_steps),
    )

    return per_beta


def update_linear_schedule(
    final_value: float,
    total_steps: int,
    initial_value: float,
    current_step: int,
):
    # learning_rate = initial_value
    if initial_value < final_value:
        clamp_func = min
    else:
        clamp_func = max
    value = clamp_func(
        final_value,
        initial_value + ((final_value - initial_value) * (current_step / total_steps)),
    )
    return value


def update_inverse_sqrt_schedule(
    initial_value: float = None,
    current_step: int = None,
):
    return initial_value / math.sqrt(current_step + 1)


def default_plot_func(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    axs[row][col].set_title(
        "{} | rolling average: {}".format(key, np.mean(values[-5:]))
    )
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, values)
    if key in targets and targets[key] is not None:
        axs[row][col].axhline(y=targets[key], color="r", linestyle="--")


def plot_scores(axs, key: str, values: list[dict], targets: dict, row: int, col: int):
    if len(values) == 0:
        return
    print(values)
    scores = [value["score"] for value in values]
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, scores)

    has_max_scores = "max_score" in values[0]
    has_min_scores = "min_score" in values[0]
    assert (
        has_max_scores == has_min_scores
    ), "Both max_scores and min_scores must be provided or not provided"

    if has_max_scores:
        max_scores = [value["max_score"] for value in values]
        min_scores = [value["min_score"] for value in values]
        axs[row][col].fill_between(x, min_scores, max_scores, alpha=0.5)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(scores[-5:])} | latest: {scores[-1]}"
    )

    axs[row][col].set_xlabel("Game")
    axs[row][col].set_ylabel("Score")

    axs[row][col].set_xlim(1, len(values))

    if len(scores) > 1:
        best_fit_x, best_fit_y = np.polyfit(x, scores, 1)
        axs[row][col].plot(
            x,
            best_fit_x * x + best_fit_y,
            color="g",
            label="Best Fit Line",
            linestyle="dotted",
        )

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_loss(axs, key: str, values: list[dict], targets: dict, row: int, col: int):
    loss = [value["loss"] for value in values]
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, loss)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(loss[-5:])} | latest: {loss[-1]}"
    )

    axs[row][col].set_xlabel("Time Step")
    axs[row][col].set_ylabel("Loss")

    axs[row][col].set_xlim(1, len(values))

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_exploitability(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    if len(values) == 0:
        return
    exploitability = [abs(value["exploitability"]) for value in values]
    print(values)
    rolling_averages = [
        np.mean(exploitability[max(0, i - 5) : i])
        for i in range(1, len(exploitability) + 1)
    ]
    # print(rolling_averages)
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, rolling_averages)
    axs[row][col].plot(x, exploitability)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    if len(rolling_averages) > 1:
        best_fit_x, best_fit_y = np.polyfit(x, rolling_averages, 1)
        axs[row][col].plot(
            x,
            best_fit_x * x + best_fit_y,
            color="g",
            label="Best Fit Line",
            linestyle="dotted",
        )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(exploitability[-5:])} | latest: {exploitability[-1]}"
    )

    axs[row][col].set_xlabel("Game")
    axs[row][col].set_ylabel("Exploitability (rolling average)")

    axs[row][col].set_xscale("log")
    axs[row][col].set_yscale("log")

    axs[row][col].set_xlim(1, len(values))
    # axs[row][col].set_ylim(0.01, 10)
    # axs[row][col].set_ylim(
    #     -(10 ** math.ceil(math.log10(abs(min_exploitability)))),
    #     10 ** math.ceil(math.log10(max_exploitability)),
    # )

    # axs[row][col].set_yticks(
    #     [
    #         -(10**i)
    #         for i in range(
    #             math.ceil(math.log10(abs(min_exploitability))),
    #             math.floor(math.log10(abs(min_exploitability))) - 1,
    #             -1,
    #         )
    #         if -(10**i) < min_exploitability
    #     ]
    #     + [0]
    #     + [
    #         10**i
    #         for i in range(
    #             math.ceil(math.log10(max_exploitability)),
    #             math.floor(math.log10(max_exploitability)) + 1,
    #         )
    #         if 10**i > max_exploitability
    #     ]
    # )

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Exploitability: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_trials(scores: list, file_name: str, final_trial: int = 0):
    fig, axs = plt.subplots(
        1,
        1,
        figsize=(10, 5),
        squeeze=False,
    )
    if final_trial > 0:
        x = np.arange(1, final_trial + 1)
        scores = scores[:final_trial]
    else:
        x = np.arange(1, len(scores) + 1)
    axs[0][0].scatter(x, scores)
    best_fit_x, best_fit_y = np.polyfit(x, scores, 1)
    axs[0][0].plot(
        x,
        best_fit_x * x + best_fit_y,
        color="g",
        label="Best Fit Line",
        linestyle="dotted",
    )

    fig.suptitle("Score of Hyperopt trials over time for Rainbow DQN on CartPole-v1")
    axs[0][0].set_xlabel("Trial")
    axs[0][0].set_ylabel("Score")
    plt.savefig(f"./graphs/{file_name}.png")
    plt.show()
    plt.close(fig)


stat_keys_to_plot_funcs = {
    "test_score": plot_scores,
    "score": plot_scores,
    "policy_loss": plot_loss,
    "value_loss": plot_loss,
    "l2_loss": plot_loss,
    "loss": plot_loss,
    "rl_loss": plot_loss,
    "sl_loss": plot_loss,
    "exploitability": plot_exploitability,  # should this be plot_scores?
}


def plot_graphs(
    stats: dict,
    targets: dict,
    step: int,
    frames_seen: int,
    time_taken: float,
    model_name: str,
    dir: str = "./checkpoints/graphs",
):
    num_plots = len(stats)
    sqrt_num_plots = math.ceil(np.sqrt(num_plots))
    fig, axs = plt.subplots(
        sqrt_num_plots,
        sqrt_num_plots,
        figsize=(10 * sqrt_num_plots, 5 * sqrt_num_plots),
        squeeze=False,
    )

    hours = int(time_taken // 3600)
    minutes = int((time_taken % 3600) // 60)
    seconds = int(time_taken % 60)

    fig.suptitle(
        "training stats | training step {} | frames seen {} | time taken {} hours {} minutes {} seconds".format(
            step, frames_seen, hours, minutes, seconds
        )
    )

    for i, (key, values) in enumerate(stats.items()):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots

        if key in stat_keys_to_plot_funcs:
            stat_keys_to_plot_funcs[key](axs, key, values, targets, row, col)
        else:
            default_plot_func(axs, key, values, targets, row, col)

    for i in range(num_plots, sqrt_num_plots**2):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        fig.delaxes(axs[row][col])

    # plt.show()
    assert os.path.exists(dir), f"Directory {dir} does not exist"
    plt.savefig("{}/{}.png".format(dir, model_name))

    plt.close(fig)


def plot_comparisons(
    stats: list[dict],
    model_name: str,
    dir: str = "./checkpoints/graphs",
):
    num_plots = len(stats[0])
    sqrt_num_plots = math.ceil(np.sqrt(num_plots))
    fig, axs = plt.subplots(
        sqrt_num_plots,
        sqrt_num_plots,
        figsize=(10 * sqrt_num_plots, 5 * sqrt_num_plots),
        squeeze=False,
    )

    fig.suptitle("Comparison of training stats")

    for i, (key, _) in enumerate(stats[0].items()):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        # max_value = float("-inf")
        # min_value = float("inf")
        max_len = 0
        for s in stats:
            values = s[key]
            # print(values)
            max_len = max(max_len, len(values))
            print(max_len)
            # max_value = max(max_value, max(values))
            # min_value = min(min_value, min(values))
            if key in stat_keys_to_plot_funcs:
                stat_keys_to_plot_funcs[key](axs, key, values, {}, row, col)
                axs[row][col].set_xlim(0, max_len)
            else:
                default_plot_func(axs, key, values, {}, row, col)

        # axs[row][col].set_ylim(min_value, max_value)

    for i in range(num_plots, sqrt_num_plots**2):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        fig.delaxes(axs[row][col])

    # plt.show()
    os.makedirs(dir, exist_ok=True)
    plt.savefig("{}/{}.png".format(dir, model_name))

    plt.close(fig)


def prepare_kernel_initializers(kernel_initializer: str, output_layer: bool = False):
    if kernel_initializer == "pytorch_default":
        return None
    if kernel_initializer == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif kernel_initializer == "glorot_normal":
        return nn.init.xavier_normal_
    elif kernel_initializer == "he_uniform":
        return nn.init.kaiming_uniform_
    elif kernel_initializer == "he_normal":
        return nn.init.kaiming_normal_
    elif kernel_initializer == "variance_baseline":
        return VarianceScaling()
    elif kernel_initializer == "variance_0.1":
        return VarianceScaling(scale=0.1)
    elif kernel_initializer == "variance_0.3":
        return VarianceScaling(scale=0.3)
    elif kernel_initializer == "variance_0.8":
        return VarianceScaling(scale=0.8)
    elif kernel_initializer == "variance_3":
        return VarianceScaling(scale=3)
    elif kernel_initializer == "variance_5":
        return VarianceScaling(scale=5)
    elif kernel_initializer == "variance_10":
        return VarianceScaling(scale=10)
    # TODO
    # elif kernel_initializer == "lecun_uniform":
    #     return LecunUniform(seed=np.random.seed())
    # elif kernel_initializer == "lecun_normal":
    #     return LecunNormal(seed=np.random.seed())
    elif kernel_initializer == "orthogonal":
        return nn.init.orthogonal_

    raise ValueError(f"Invalid kernel initializer: {kernel_initializer}")


def prepare_activations(activation: str):
    # print("Activation to prase: ", activation)
    if activation == "linear":
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "soft_sign":
        return nn.Softsign()
    elif activation == "silu" or activation == "swish":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    # elif activation == "log_sigmoid":
    #     return nn.LogSigmoid()
    elif activation == "hard_sigmoid":
        return nn.Hardsigmoid()
    # elif activation == "hard_silu" or activation == "hard_swish":
    #     return nn.Hardswish()
    # elif activation == "hard_tanh":
    #     return nn.Hardtanh()
    elif activation == "elu":
        return nn.ELU()
    # elif activation == "celu":
    #     return nn.CELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "gelu":
        return nn.GELU()
    # elif activation == "glu":
    #     return nn.GLU()

    raise ValueError(f"Activation {activation} not recognized")


def epsilon_greedy_policy(
    q_values: list[float], info: dict, epsilon: float, wrapper=np.argmax
):
    if np.random.rand() < epsilon:
        # print("selecting a random move")
        if "legal_moves" in info:
            # print("using legal moves")
            return random.choice(info["legal_moves"])
        else:
            q_values = q_values.reshape(-1)
            return random.choice(range(len(q_values)))
    else:
        # try:
        # print("using provided wrapper to select action")
        return wrapper(q_values, info)
    # except:
    #     return wrapper(q_values)


def add_dirichlet_noise(
    policy: list[float], dirichlet_alpha: float, exploration_fraction: float
):
    # MAKE ALPHAZERO USE THIS
    noise = np.random.dirichlet([dirichlet_alpha] * len(policy))
    frac = exploration_fraction
    for i, n in enumerate(noise):
        policy[i] = (1 - frac) * policy[i] + frac * n
    return policy


def augment_game(game, flip_y: bool = False, flip_x: bool = False, rot90: bool = False):
    # augmented_games[0] = rotate 90
    # augmented_games[1] = rotate 180
    # augmented_games[2] = rotate 270
    # augmented_games[3] = flip y (rotate 180 and flip x)
    # augmented_games[4] = rotate 90 and flip y (rotate 270 and flip x)
    # augmented_games[5] = rotate 180 and flip y (flip x)
    # augmented_games[6] = flip y and rotate 90 (rotate 270 and flip y) (rotate 90 and flip x)
    # augmented_games[7] = normal

    if (rot90 and flip_y) or (rot90 and flip_x):
        augemented_games = [copy.deepcopy(game) for _ in range(7)]
        for i in range(len(game.observation_history)):
            board = game.observation_history[i]
            policy = game.policy_history[i]
            augemented_games[0].observation_history[i] = np.rot90(board)
            augemented_games[0].policy_history[i] = np.rot90(policy)
            augemented_games[1].observation_history[i] = np.rot90(np.rot90(board))
            augemented_games[1].policy_history[i] = np.rot90(np.rot90(policy))
            augemented_games[2].observation_history[i] = np.rot90(
                np.rot90(np.rot90(board))
            )
            augemented_games[2].policy_history[i] = np.rot90(np.rot90(np.rot90(policy)))
            augemented_games[3].observation_history[i] = np.flipud(board)
            augemented_games[3].policy_history[i] = np.flipud(policy)
            augemented_games[4].observation_history[i] = np.flipud(np.rot90(board))
            augemented_games[4].policy_history[i] = np.flipud(np.rot90(policy))
            augemented_games[5].observation_history[i] = np.flipud(
                np.rot90(np.rot90(board))
            )
            augemented_games[5].policy_history[i] = np.flipud(
                np.rot90(np.rot90(policy))
            )
            augemented_games[6].observation_history[i] = np.rot90(np.flipud(board))
            augemented_games[6].policy_history[i] = np.rot90(np.flipud(policy))
    elif rot90 and not flip_y and not flip_x:
        augemented_games = [copy.deepcopy(game) for _ in range(3)]
        augemented_games[0].observation_history = [
            np.rot90(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.rot90(policy) for policy in game.policy_history
        ]
        augemented_games[1].observation_history = [
            np.rot90(np.rot90(board)) for board in game.observation_history
        ]
        augemented_games[1].policy_history = [
            np.rot90(np.rot90(policy)) for policy in game.policy_history
        ]
        augemented_games[2].observation_history = [
            np.rot90(np.rot90(np.rot90(board))) for board in game.observation_history
        ]
        augemented_games[2].policy_history = [
            np.rot90(np.rot90(np.rot90(policy)) for policy in game.policy_history)
        ]
    elif flip_y and not rot90 and not flip_x:
        augemented_games = [copy.deepcopy(game)]
        augemented_games[0].observation_history = [
            np.flipud(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.flipud(policy) for policy in game.policy_history
        ]

    elif flip_x and not rot90 and not flip_y:
        augemented_games = [copy.deepcopy(game) for _ in range(1)]
        augemented_games[0].observation_history = [
            np.fliplr(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.fliplr(policy) for policy in game.policy_history
        ]

    augemented_games.append(game)
    return augemented_games


def augment_board(
    board, policy, flip_y: bool = False, flip_x: bool = False, rot90: bool = False
):
    if (rot90 and flip_y) or (rot90 and flip_x):
        augemented_boards = [copy.deepcopy(board) for _ in range(7)]
        augmented_policies = [copy.deepcopy(policy) for _ in range(7)]
        augemented_boards[0] = np.rot90(board)
        augmented_policies[0] = np.rot90(policy)
        augemented_boards[1] = np.rot90(np.rot90(board))
        augmented_policies[1] = np.rot90(np.rot90(policy))
        augemented_boards[2] = np.rot90(np.rot90(np.rot90(board)))
        augmented_policies[2] = np.rot90(np.rot90(np.rot90(policy)))
        augemented_boards[3] = np.flipud(board)
        augmented_policies[3] = np.flipud(policy)
        augemented_boards[4] = np.flipud(np.rot90(board))
        augmented_policies[4] = np.flipud(np.rot90(policy))
        augemented_boards[5] = np.flipud(np.rot90(np.rot90(board)))
        augmented_policies[5] = np.flipud(np.rot90(np.rot90(policy)))
        augemented_boards[6] = np.rot90(np.flipud(board))
        augmented_policies[6] = np.rot90(np.flipud(policy))
    elif rot90 and not flip_y and not flip_x:
        augemented_boards = [copy.deepcopy(board) for _ in range(3)]
        augmented_policies = [copy.deepcopy(policy) for _ in range(3)]
        augemented_boards[0] = np.rot90(board)
        augmented_policies[0] = np.rot90(policy)
        augemented_boards[1] = np.rot90(np.rot90(board))
        augmented_policies[1] = np.rot90(np.rot90(policy))
        augemented_boards[2] = np.rot90(np.rot90(np.rot90(board)))
        augmented_policies[2] = np.rot90(np.rot90(np.rot90(policy)))
    elif flip_y and not rot90 and not flip_x:
        augemented_boards = [copy.deepcopy(board)]
        augmented_policies = [copy.deepcopy(policy)]
        augemented_boards[0] = np.flipud(board)
        augmented_policies[0] = np.flipud(policy)
    elif flip_x and not rot90 and not flip_y:
        augemented_boards = [copy.deepcopy(board)]
        augmented_policies = [copy.deepcopy(policy)]
        augemented_boards[0] = np.fliplr(board)
        augmented_policies[0] = np.fliplr(policy)
    augemented_boards.append(board)
    augmented_policies.append(policy)
    return augemented_boards, augmented_policies


def sample_by_random_indices(
    max_index_or_1darray, batch_size: int, with_replacement=False
) -> npt.NDArray[np.int64]:
    """
    Sample from a numpy array using indices
    """
    return np.random.choice(max_index_or_1darray, batch_size, replace=with_replacement)


def sample_by_indices_probability(
    max_index_or_1darray, batch_size: int, probabilities: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """
    Sample from a numpy array using indices
    """
    return np.random.choice(max_index_or_1darray, batch_size, p=probabilities)


def sample_tree_proportional(
    tree, batch_size: int, max_size: int
) -> npt.NDArray[np.int64]:
    """
    tree: SumSegmentTree
    Sample proportionally from a sum segment tree. Used in prioritized experience replay
    """
    indices = np.zeros(batch_size, dtype=np.int64)
    total_priority = tree.sum(0, max_size - 1)
    priority_segment = total_priority / batch_size

    for i in range(batch_size):
        l = priority_segment * i
        h = priority_segment * (i + 1)
        upperbound = np.random.uniform(l, h)
        indices[i] = tree.retrieve(upperbound)
        # print(tree[indices[i]])

    return indices


def reward_clipping(reward: float, lower_bound: float = -1, upper_bound: float = 1):
    if reward < lower_bound:
        return lower_bound
    elif reward > upper_bound:
        return upper_bound
    return reward


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def to_lists(l: list[Iterable]) -> list[Tuple]:
    """Convert a list of iterables to a zip of tuples

    Args:
        list (list[Iterable]): A list of iterables, e.g. [(1,1,1),(2,2,2),(3,3,3)]

    Returns:
        list[Tuple]: A list of tuples, i.e. [(1,2,3), (1,2,3), (1,2,3)]
    """

    return list(zip(*l))


def current_timestamp():
    return datetime.now().timestamp()


_epsilon = 1e-7


def categorical_crossentropy(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    # print(predicted)
    predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    # print(predicted)
    predicted = torch.clamp(predicted, _epsilon, 1.0 - _epsilon)
    # print(predicted)
    log_prob = torch.log(predicted)
    return -torch.sum(log_prob * target, axis=axis)


class CategoricalCrossentropyLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return categorical_crossentropy(predicted, target, self.axis)


def kl_divergence(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    predicted = torch.clamp(predicted, _epsilon, 1.0)
    target = torch.clamp(target, _epsilon, 1.0)
    return torch.sum(target * torch.log(target / predicted), axis=axis)


class KLDivergenceLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return kl_divergence(predicted, target, self.axis)


def huber(predicted: torch.Tensor, target: torch.Tensor, axis=-1, delta: float = 1.0):
    diff = torch.abs(predicted - target)
    return torch.where(
        diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)
    ).view(-1)


class HuberLoss:
    def __init__(self, axis=-1, delta: float = 1.0):
        self.axis = axis
        self.delta = delta

    def __call__(self, predicted, target):
        return huber(predicted, target, axis=self.axis, delta=self.delta)


def mse(predicted: torch.Tensor, target: torch.Tensor):
    # print(predicted)
    # print(target)
    return (predicted - target) ** 2


class MSELoss:
    def __init__(self):
        pass

    def __call__(self, predicted, target):
        return mse(predicted, target)


from typing import Callable

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def calculate_padding(i: int, k: int, s: int) -> Tuple[int, int]:
    """Calculate both padding sizes along 1 dimension for a given input length, kernel length, and stride

    Args:
        i (int): input length
        k (int): kernel length
        s (int): stride

    Returns:
        (p_1, p_2): where p_1 = p_2 - 1 for uneven padding and p_1 == p_2 for even padding
    """

    p = (i - 1) * s - i + k
    p_1 = p // 2
    p_2 = (p + 1) // 2
    return (p_1, p_2)


def generate_layer_widths(widths: list[int], max_num_layers: int) -> list[Tuple[int]]:
    """Create all possible combinations of widths for a given number of layers"""
    width_combinations = []

    for i in range(0, max_num_layers):
        width_combinations.extend(itertools.combinations_with_replacement(widths, i))

    return width_combinations


def hyperopt_analysis(
    data_dir: str,
    file_name: str,
    viable_trial_threshold: int,
    step: int,
    final_trial: int = 0,
    eval_method: str = "final_score",
):
    trials = pickle.load(open(f"{data_dir}/{file_name}.p", "rb"))
    if final_trial > 0:
        print("Number of trials: {}".format(final_trial))
    else:
        print("Number of trials: {}".format(len(trials.trials)))
    # losses.sort()
    # print(len(os.listdir(f"{data_dir}/checkpoints")) - 1)
    # print(len(trials.trials))

    checkpoints = os.listdir(f"{data_dir}/checkpoints")
    checkpoints.remove("videos") if "videos" in checkpoints else None
    checkpoints.remove(".DS_Store") if ".DS_Store" in checkpoints else None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    if final_trial > 0:
        checkpoints = checkpoints[:final_trial]

    viable_throughout_trials = []
    final_rolling_averages = []
    final_std_devs = []
    scores = []
    losses = []
    failed_trials = 0
    for i, trial in enumerate(trials.trials):
        losses.append(trial["result"]["loss"])
        if final_trial > 0 and i >= final_trial:
            break
        # print(trial["result"]["status"])
        if trial["result"]["status"] == "fail":
            failed_trials += 1
            final_rolling_averages.append(trial["result"]["loss"])
            scores.append(trial["result"]["loss"])
            final_std_devs.append(trial["result"]["loss"])
        else:
            # print(checkpoints[i - failed_trials])
            # print(failed_trials)
            # if os.path.exists(
            #     f"{data_dir}/checkpoints/{checkpoints[i - failed_trials]}/step_{step}/graphs_stats/stats.pkl"
            # ):
            stats = pickle.load(
                open(
                    f"{data_dir}/checkpoints/{checkpoints[i - failed_trials]}/step_{step}/graphs_stats/stats.pkl",
                    "rb",
                )
            )
            max_score = 0

            # print([stat_dict["score"] for stat_dict in stats["test_score"][-5:]])
            final_rolling_averages.append(
                np.around(
                    np.mean(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-5:]]
                    ),
                    1,
                )
            )

            final_std_devs.append(
                np.around(
                    np.std(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-5:]]
                    ),
                    1,
                )
            )

            for stat_dict in stats["test_score"]:
                if stat_dict["max_score"] > max_score:
                    max_score = stat_dict["max_score"]

            if max_score > viable_trial_threshold:
                viable_throughout_trials.append(max_score)

            if eval_method == "final_score":
                score = -trial["result"]["loss"]
            elif (
                eval_method == "rolling_average"
                or eval_method == "final_score_rolling_average"
            ):
                score = stats["test_score"][-1]["score"]
            scores.append(score)

    plot_trials(
        scores,
        file_name,
        final_trial=final_trial,
    )

    res = [
        list(x)
        for x in zip(
            *sorted(
                zip(losses, scores, final_rolling_averages, final_std_devs),
                key=itemgetter(0),
            )
        )
    ]
    losses = res[0]
    scores = res[1]
    final_rolling_averages = res[2]
    final_std_devs = res[3]
    viable_trials = [score for score in scores if score > viable_trial_threshold]

    print("Failed trials: ~{}%".format(round(failed_trials / len(scores) * 100)))

    print(
        "Viable trials (based on final score): ~{}%".format(
            round(len(viable_trials) / len(scores) * 100)
        )
    )
    print(
        "Viable trials (throughout training): ~{}%".format(
            round(len(viable_throughout_trials) / len(scores) * 100)
        )
    )

    print("Losses: {}".format(losses))
    print("Scores: {}".format(scores))
    print("Final rolling averages: {}".format(final_rolling_averages))
    print("Final standard deviations: {}".format(final_std_devs))

    print("Max loss: {}".format(max(losses)))
    print("Max score: {}".format(max(scores)))
    print("Max final rolling average: {}".format(max(final_rolling_averages)))
    print("Max final standard deviation: {}".format(max(final_std_devs)))

    print("Average loss: {}".format(np.mean(losses)))
    print("Average score: {}".format(np.mean(scores)))
    print("Average final rolling average: {}".format(np.mean(final_rolling_averages)))
    print("Average final standard deviation: {}".format(np.mean(final_std_devs)))

    viable_final_rolling_averages = [
        final_rolling_averages[i]
        for i, loss in enumerate(scores)
        if loss > viable_trial_threshold
    ]

    viable_std_devs = [
        final_std_devs[i]
        for i, loss in enumerate(scores)
        if loss > viable_trial_threshold
    ]

    print(
        "Average score of viable trials (based on final score): {}".format(
            np.mean(viable_trials)
        )
    )
    print(
        "Average final rolling average of viable trials (based on final score): {}".format(
            np.mean(viable_final_rolling_averages)
        )
    )
    print(
        "Average final standard deviation of viable trials (based on final score): {}".format(
            np.mean(viable_std_devs)
        )
    )


def graph_hyperparameter_importance(
    data_dir: str, trials_file: str, search_space_file: str, viable_trial_threshold: int
):
    with open(f"{data_dir}/{trials_file}", "rb") as f:
        trials = pickle.load(f)
    print(trials)

    search_space = pickle.load(open(f"./search_spaces/{search_space_file}", "rb"))

    values_dict = defaultdict(list)
    scores = []
    for trial in trials.trials:
        for key, value in space_eval(trial["misc"]["vals"], search_space).items():
            values_dict[key].append(value[0])
        scores.append(-trial["result"]["loss"])

    df = pd.DataFrame(values_dict)
    x_cols = df.columns
    df["scores"] = scores
    # print(df)
    df = df[df["scores"] > viable_trial_threshold]

    for col in x_cols:
        if col == "loss_function":
            continue
        plt = df.plot(x=col, y="scores", kind="scatter")
        grouped = df.groupby(col)["scores"]
        medians = grouped.median()
        means = grouped.mean()
        stddev = grouped.std()

        if not (col == "kernel_initializer" or col == "activation"):
            # plt.fill_between(medians.index, medians.values-stddev, medians.values+stddev, color="#00F0F0")
            plt.plot(means.index, means.values, color="#00FFFF")
        else:
            plt.scatter(means.index, means.values, c="#00FFFF")
        # plt.add_line


def calc_units(shape):
    shape = tuple(shape)
    if len(shape) == 1:
        return shape + shape
    if len(shape) == 2:
        # dense layer -> (in_channels, out_channels)
        return shape
    else:
        # conv_layer (Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (input_depth, depth, ...)
        in_units = shape[1]
        out_units = shape[0]
        c = 1
        for dim in shape[2:]:
            c *= dim
        return (c * in_units, c * out_units)


class VarianceScaling:
    def __init__(self, scale=0.1, mode="fan_in", distribution="uniform"):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

        assert mode == "fan_in" or mode == "fan_out" or mode == "fan_avg"
        assert distribution == "uniform", "only uniform distribution is supported"

    def __call__(self, tensor: Tensor) -> None:
        with torch.no_grad():
            scale = self.scale
            shape = tensor.shape
            in_units, out_units = calc_units(shape)
            if self.mode == "fan_in":
                scale /= in_units
            elif self.mode == "fan_out":
                scale /= out_units
            else:
                scale /= (in_units + out_units) / 2

            limit = math.sqrt(3.0 * scale)
            return tensor.uniform_(-limit, limit)


def isiterable(o):
    try:
        it = iter(o)
    except TypeError:
        return False
    return True


def tointlists(list):
    ret = []
    for x in list:
        if isiterable(x):
            ret.append(tointlists(x))
        else:
            ret.append(int(x))
    return ret


import time
from collections import deque


class StoppingCriteria:
    def __init__(self):
        pass

    def should_stop(self, details: dict) -> bool:
        return False


class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, max_runtime_sec=60 * 10):
        self.stop_time = time.time() + max_runtime_sec

    def should_stop(self, details: dict) -> bool:
        return time.time() > self.stop_time


class TrainingStepStoppingCritiera(StoppingCriteria):
    def __init__(self, max_training_steps=100000):
        self.max_training_steps = max_training_steps

    def should_stop(self, details: dict) -> bool:
        return details["training_step"] > self.max_training_steps


class EpisodesStoppingCriteria(StoppingCriteria):
    def __init__(self, max_episodes=100000):
        self.max_episodes = max_episodes

    def should_stop(self, details: dict) -> bool:
        return details["max_episodes"] > self.max_episodes


class AverageScoreStoppingCritera(StoppingCriteria):
    def __init__(self, min_avg_score: float, last_scores_length: int):
        self.min_avg_score = min_avg_score
        self.last_scores_length = last_scores_length
        self.last_scores = deque(maxlen=last_scores_length)

    def add_score(self, score: float):
        self.last_scores.append(score)

    def should_stop(self, details: dict) -> bool:
        if len(self.last_scores) < self.last_scores_length:
            return False

        return np.average(self.last_scores) < self.min_avg_score


class ApexLearnerStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.criterias: dict[str, StoppingCriteria] = {
            "time": TimeStoppingCriteria(max_runtime_sec=1.5 * 60 * 60),
            "training_step": TrainingStepStoppingCritiera(max_training_steps=10000),
            "avg_score": AverageScoreStoppingCritera(
                min_avg_score=15, last_scores_length=10
            ),
        }

    def should_stop(self, details: dict) -> bool:
        if self.criterias["time"].should_stop(details):
            return True

        if details["training_step"] < 10000:
            return False

        return self.criterias["training_step"].should_stop(details) or self.criterias[
            "avg_score"
        ].should_stop(details)

    def add_score(self, score: float):
        tc: AverageScoreStoppingCritera = self.criterias["avg_score"]
        tc.add_score(score)


import gc
import os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import copy
import pickle
from torch.optim import Optimizer
from torch.nn import Module

from utils import make_stack, plot_graphs

# Every model should have:
# 1. A network
# 2. An optimizer
# 3. A loss function
# 4. A training method
#       this method should have training iterations, minibatches, and training steps
# 6. A select_action method
# 7. A predict method

import torch
import yaml


class ConfigBase:
    def parse_field(
        self, field_name, default=None, wrapper=None, required=True, dtype=None
    ):
        if field_name in self.config_dict:
            val = self.config_dict[field_name]
            # print("value: ", val)
            print(f"Using         {field_name:30}: {val}")
            if wrapper is not None:
                return wrapper(val)
            return self.config_dict[field_name]

        if default is not None:
            print(f"Using default {field_name:30}: {default}")
            if wrapper is not None:
                return wrapper(default)
            return default

        if required:
            raise ValueError(
                f"Missing required field without default value: {field_name}"
            )
        else:
            print(f"Using         {field_name:30}: {default}")

        if field_name in self._parsed_fields:
            print("warning: duplicate field: ", field_name)
        self._parsed_fields.add(field_name)

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self._parsed_fields = set()

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            print(o)
            a = cls(config_dict=o["config_dict"])

        return a

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict)

        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)


class GameConfig:
    def __init__(
        self,
        max_score,
        min_score,
        is_discrete,
        is_image,
        is_deterministic,
        has_legal_moves,
        perfect_information,
        multi_agent,
        num_players,
    ):
        self.max_score = max_score
        self.min_score = min_score
        self.is_discrete = is_discrete  # can just check the action space type instead of setting manually if the env is passed in (ALSO COULD DO THIS IN THE BASE GAME CONFIG)
        # self.num_actions = num_actions
        # self.observation_space = observation_space
        self.is_image = is_image
        self.is_deterministic = is_deterministic
        # self.num_players = num_players (might not need this idk) <- it would likely be for muzero but could also be for rainbow and stuff when they play multiplayer games (like connect 4)
        self.has_legal_moves = has_legal_moves
        self.perfect_information = perfect_information
        self.multi_agent = multi_agent
        self.num_players = num_players

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, GameConfig):
            return False

        return (
            self.max_score == o.max_score
            and self.min_score == o.min_score
            and self.is_discrete == o.is_discrete
            and self.is_image == o.is_image
            and self.is_deterministic == o.is_deterministic
            and self.has_legal_moves == o.has_legal_moves
            and self.perfect_information == o.perfect_information
            and self.multi_agent == o.multi_agent
            and self.num_players == o.num_players
        )


class AtariConfig(GameConfig):
    def __init__(self):
        super(AtariConfig, self).__init__(
            max_score=10,  # FROM CATEGORICAL DQN PAPER
            min_score=-10,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,  # if no frameskip, then deterministic
            has_legal_moves=False,
            perfect_information=True,  # although it is not deterministic, it is so close to it that it is considered perfect information
            multi_agent=False,
            num_players=1,
        )


class Config(ConfigBase):
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            print(o)
            a = cls(config_dict=o["config_dict"], game_config=o["game"])

        return a

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict, game=self.game)

        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)

    def __init__(self, config_dict: dict, game_config: GameConfig) -> None:
        super().__init__(config_dict)
        # could take in a game config and set an action space and observation shape here
        # OR DO THAT IN BASE AGENT?
        self.game = game_config

        self._verify_game()

        # not hyperparameters but utility things
        self.save_intermediate_weights: bool = self.parse_field(
            "save_intermediate_weights", False
        )

        # ADD LEARNING RATE SCHEDULES
        self.training_steps: int = self.parse_field(
            "training_steps", 10000, wrapper=int
        )

        self.adam_epsilon: float = self.parse_field("adam_epsilon", 1e-6)
        self.momentum = self.parse_field("momentum", 0.9)
        self.learning_rate: float = self.parse_field("learning_rate", 0.001)
        self.clipnorm: int = self.parse_field("clipnorm", 0)
        self.optimizer: torch.optim.Optimizer = self.parse_field(
            "optimizer", torch.optim.Adam
        )
        self.weight_decay: float = self.parse_field("weight_decay", 0.0)
        self.loss_function = self.parse_field("loss_function", required=True)
        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field(
            "kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )

        self.minibatch_size: int = self.parse_field("minibatch_size", 64, wrapper=int)
        self.replay_buffer_size: int = self.parse_field(
            "replay_buffer_size", 5000, wrapper=int
        )
        self.min_replay_buffer_size: int = self.parse_field(
            "min_replay_buffer_size", self.minibatch_size, wrapper=int
        )
        self.num_minibatches: int = self.parse_field("num_minibatches", 1, wrapper=int)
        self.training_iterations: int = self.parse_field(
            "training_iterations", 1, wrapper=int
        )


    def _verify_game(self):
        raise NotImplementedError


class BaseAgent:
    def __init__(
        self,
        env: gym.Env,
        config: Config,
        name,
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
        from_checkpoint=False,
    ):
        if from_checkpoint:
            self.from_checkpoint = True

        self.model: Module = None
        self.optimizer: Optimizer = None
        self.model_name = name
        self.config = config
        self.device = device

        self.training_time = 0
        self.training_step = 0
        self.total_environment_steps = 0
        self.training_steps = self.config.training_steps
        self.checkpoint_interval = max(self.training_steps // 30, 1)
        self.checkpoint_trials = 5

        self.env = env
        self.test_env = self.make_test_env(env)
        self.observation_dimensions = self.determine_observation_dimensions(env)

        print("observation_dimensions: ", self.observation_dimensions)
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = env.action_space.n
            self.discrete_action_space = True
        else:
            self.num_actions = env.action_space.shape[0]
            self.discrete_action_space = False

        print("num_actions: ", self.num_actions)

    def make_test_env(self, env: gym.Env):
        # self.test_env = copy.deepcopy(env)
        if hasattr(env, "render_mode") and env.render_mode == "rgb_array":
            # assert (
            #     self.env.render_mode == "rgb_array"
            # ), "Video recording for test_env requires render_mode to be 'rgb_array'"
            return gym.wrappers.RecordVideo(
                copy.deepcopy(env),
                ".",
                name_prefix="{}".format(self.model_name),
            )
        else:
            print(
                "Warning: test_env will not record videos as render_mode is not 'rgb_array'"
            )
            return copy.deepcopy(env)

    def determine_observation_dimensions(self, env: gym.Env):
        if isinstance(env.observation_space, gym.spaces.Box):
            return torch.Size(env.observation_space.shape)
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            return torch.Size((1,))
        elif isinstance(env.observation_space, gym.spaces.Tuple):
            return torch.Size((len(env.observation_space.spaces),))  # for tuple of discretes
        else:
            raise ValueError("Observation space not supported")

    def train(self):
        if self.training_steps != 0:
            self.print_resume_training()

        pass

    def preprocess(self, states) -> torch.Tensor:
        """Applies necessary preprocessing steps to a batch of environment observations or a single environment observation
        Does not alter the input state parameter, instead creating a new Tensor on the inputted device (default cpu)

        Args:
            state (Any): A or a list of state returned from self.env.step
        Returns:
            Tensor: The preprocessed state, a tensor of floats. If the input was a single environment step,
                    the returned tensor is returned as outputed as if a batch of states with a length of a batch size of 1
        """

        # always convert to np.array first for performance, recoommnded by pytorchx
        # special case: list of compressed images (which are LazyFrames)
        if isinstance(states[0], gym.wrappers.frame_stack.LazyFrames):
            np_states = np.array([np.array(state) for state in states])
        else:
            # single observation, could be compressed or not compressed
            # print("Single state")
            np_states = np.array(states)

        # print("Numpyified States", np_states)
        prepared_state = (
            torch.from_numpy(
                np_states,
            )
            .to(torch.float32)
            .to(self.device)
        )
        # if self.config.game.is_image:
        # normalize_images(prepared_state)

        # if the state is a single number, add a dimension (not the batch dimension!, just wrapping it in []s basically)
        if prepared_state.shape == torch.Size([]):
            prepared_state = prepared_state.unsqueeze(0)

        if prepared_state.shape == self.observation_dimensions:
            prepared_state = make_stack(prepared_state)
        return prepared_state

    def predict(
        self, state: torch.Tensor, *args
    ) -> torch.Tensor:  # args is for info for player counts or legal move masks
        """Run inference on 1 or a batch of environment states, applying necessary preprocessing steps

        Returns:
            Tensor: The predicted values, e.g. Q values for DQN or Q distributions for Categorical DQN
        """
        raise NotImplementedError

    def select_actions(self, predicted, info, mask_actions=False) -> torch.Tensor:
        """Return actions determined from the model output, appling postprocessing steps such as masking beforehand

        Args:
            state (_type_): _description_
            legal_moves (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            Tensor: _description_
        """
        raise NotImplementedError

    def learn(self):
        # raise NotImplementedError, "Every agent should have a learn method. (Previously experience_replay)"
        pass

    def load_optimizer_state(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def load_replay_buffers(self, checkpoint):
        self.replay_buffer = checkpoint["replay_buffer"]

    def load_model_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])

    def checkpoint_base(self, checkpoint):
        checkpoint["training_time"] = self.training_time
        checkpoint["training_step"] = self.training_step
        checkpoint["total_environment_steps"] = self.total_environment_steps
        return checkpoint

    def checkpoint_environment(self, checkpoint):
        checkpoint["enviroment"] = self.env
        return checkpoint

    def checkpoint_optimizer_state(self, checkpoint):
        checkpoint["optimizer"] = self.optimizer.state_dict()
        return checkpoint

    def checkpoint_replay_buffers(self, checkpoint):
        checkpoint["replay_buffer"] = self.replay_buffer
        return checkpoint

    def checkpoint_model_weights(self, checkpoint):
        checkpoint["model"] = self.model.state_dict()
        return checkpoint

    def checkpoint_extra(self, checkpoint) -> dict:
        return checkpoint

    @classmethod
    def load(cls, *args, **kwargs):
        cls.loaded_from_checkpoint = True
        return cls.load_from_checkpoint(*args, **kwargs)

    def load_from_checkpoint(agent_class, config_class, dir: str, training_step):
        # load the config and checkpoint
        training_step_dir = Path(dir, f"step_{training_step}")
        weights_dir = Path(training_step_dir, "model_weights")
        weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
        config = config_class.load(Path(dir, "configs/config.yaml"))
        checkpoint = torch.load(weights_path)
        env = checkpoint["enviroment"]
        model_name = checkpoint["model_name"]

        # construct the agent
        agent = agent_class(env, config, model_name, from_checkpoint=True)

        # load the model state (weights, optimizer, replay buffer, training time, training step, total environment steps)
        os.makedirs(weights_dir, exist_ok=True)

        agent.training_time = checkpoint["training_time"]
        agent.training_step = checkpoint["training_step"]
        agent.total_environment_steps = checkpoint["total_environment_steps"]

        agent.load_model_weights(checkpoint)
        agent.load_optimizer_state(checkpoint)
        agent.load_replay_buffers(checkpoint)

        # load the graph stats and targets
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "rb") as f:
            agent.stats = pickle.load(f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "rb") as f:
            agent.targets = pickle.load(f)

        return agent

    def save_checkpoint(
        self,
        frames_seen=None,
        training_step=None,
        time_taken=None,
    ):
        if not frames_seen is None:
            print(
                "warning: frames_seen option is deprecated, update self.total_environment_steps instead"
            )

        if not time_taken is None:
            print(
                "warning: time_taken option is deprecated, update self.training_time instead"
            )

        if not training_step is None:
            print(
                "warning: training_step option is deprecated, update self.training_step instead"
            )

        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{self.training_step}")
        os.makedirs(dir, exist_ok=True)

        # save the model state
        if self.config.save_intermediate_weights:
            weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
            os.makedirs(Path(training_step_dir, "model_weights"), exist_ok=True)
            checkpoint = self.make_checkpoint_dict(checkpoint)
            torch.save(checkpoint, weights_path)

        if self.env.render_mode == "rgb_array":
            os.makedirs(Path(training_step_dir, "videos"), exist_ok=True)

        # save config
        os.makedirs(Path(dir, "configs"), exist_ok=True)
        self.config.dump(f"{dir}/configs/config.yaml")

        # test model
        test_score = self.test(
            self.checkpoint_trials, self.training_step, training_step_dir
        )
        self.stats["test_score"].append(test_score)
        # save the graph stats and targets
        os.makedirs(
            Path(training_step_dir, f"graphs_stats", exist_ok=True), exist_ok=True
        )
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "wb") as f:
            pickle.dump(self.targets, f)

        # to periodically clear uneeded memory, if it is drastically slowing down training you can comment this out, checkpoint less often, or do less trials
        gc.collect()

        # plot the graphs (and save the graph)
        print(self.stats)
        print(self.targets)

        os.makedirs(Path(dir, "graphs"), exist_ok=True)
        plot_graphs(
            self.stats,
            self.targets,
            self.training_step if training_step is None else training_step,
            self.total_environment_steps if frames_seen is None else frames_seen,
            self.training_time if time_taken is None else time_taken,
            self.model_name,
            f"{dir}/graphs",
        )

    def make_checkpoint_dict(self):
        checkpoint = self.checkpoint_base({})
        checkpoint = self.checkpoint_environment(checkpoint)
        checkpoint = self.checkpoint_optimizer_state(checkpoint)
        checkpoint = self.checkpoint_replay_buffers(checkpoint)
        checkpoint = self.checkpoint_model_weights(checkpoint)
        checkpoint = self.checkpoint_extra(checkpoint)
        return checkpoint

    def test(self, num_trials, step, dir="./checkpoints") -> None:
        if num_trials == 0:
            return
        with torch.no_grad():
            """Test the agent."""
            average_score = 0
            max_score = float("-inf")
            min_score = float("inf")
            # self.test_env.reset()
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}/{}".format(
                    dir, self.model_name, step
                )
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)
            for trials in range(num_trials):
                state, info = self.test_env.reset()

                done = False
                score = 0

                while not done:
                    prediction = self.predict(
                        state, info, env=self.test_env
                    )  # env = self.test_env is there for alpha_zero which needs to use the test env here instead of the normal env for the tree search (might be able to just use the regular env still)
                    action = self.select_actions(
                        prediction, info, self.config.game.has_legal_moves
                    ).item()
                    next_state, reward, terminated, truncated, info = (
                        self.test_env.step(action)
                    )
                    # self.test_env.render()
                    done = terminated or truncated
                    state = next_state
                    score += reward[0] if isinstance(reward, list) else reward
                average_score += score
                max_score = max(max_score, score)
                min_score = min(min_score, score)
                print("score: ", score)

            # reset
            # if self.test_env.render_mode != "rgb_array":
            #     self.test_env.render()
            # self.test_env.close()
            average_score /= num_trials
            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
            }

    def print_training_progress(self):
        print(f"Training step: {self.training_step + 1}/{self.training_steps}")

    def print_resume_training(self):
        print(
            f"Resuming training at step {self.training_step + 1} / {self.training_steps}"
        )

    def print_stats(self):
        print(f"")


def unpack(x: int | Tuple):
    if isinstance(x, Tuple):
        assert len(x) == 2
        return x
    else:
        try:
            x = int(x)
            return x, x
        except Exception as e:
            print(f"error converting {x} to int: ", e)


class Conv2dStack(nn.Module):
    @staticmethod
    def calculate_same_padding(i, k, s) -> Tuple[None | Tuple[int], None | str | Tuple]:
        """Calculate pytorch inputs for same padding
        Args:
            i (int, int) or int: (h, w) or (w, w)
            k (int, int) or int: (k_h, k_w) or (k, k)
            s (int, int) or int: (s_h, s_w) or (s, s)
        Returns:
            Tuple[manual_pad_padding, torch_conv2d_padding_input]: Either the manual padding that must be applied (first element of tuple) or the input to the torch padding argument of the Conv2d layer
        """

        if s == 1:
            return None, "same"
        h, w = unpack(i)
        k_h, k_w = unpack(k)
        s_h, s_w = unpack(s)
        p_h = calculate_padding(h, k_h, s_h)
        p_w = calculate_padding(w, k_w, s_w)
        if p_h[0] == p_h[1] and p_w[0] == p_w[1]:
            return None, (p_h[0], p_w[0])
        else:
            # not torch compatiable, manually pad with torch.nn.functional.pad
            return (*p_w, *p_h), None

    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
    ):
        """A sequence of convolution layers with the activation function applied after each layer.
        Always applies the minimum zero-padding that ensures the output shape is equal to the input shape.
        Input shape in "BCHW" form, i.e. (batch_size, input_channels, height, width)
        """
        super(Conv2dStack, self).__init__()
        self.conv_layers = nn.ModuleList()

        self.activation = activation

        # [B, C_in, H, W]
        assert len(input_shape) == 4
        assert len(filters) == len(kernel_sizes) == len(strides)
        assert len(filters) > 0

        self.noisy = noisy_sigma != 0
        if self.noisy:
            print("warning: Noisy convolutions not implemented yet")
            # raise NotImplementedError("")

        current_input_channels = input_shape[1]
        for i in range(len(filters)):

            h, w = input_shape[2], input_shape[3]
            manual_padding, torch_padding = self.calculate_same_padding(
                (h, w), kernel_sizes[i], strides[i]
            )

            if not torch_padding is None:
                layer = nn.Conv2d(
                    in_channels=current_input_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=torch_padding,
                )
            else:
                layer = nn.Sequential(
                    nn.ZeroPad2d(manual_padding),
                    nn.Conv2d(
                        in_channels=current_input_channels,
                        out_channels=filters[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                    ),
                )

            self.conv_layers.append(layer)
            current_input_channels = filters[i]

        self._output_len = current_input_channels

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        def initialize_if_conv(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)

        self.apply(initialize_if_conv)

    def forward(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    def remove_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    @property
    def output_channels(self):
        return self._output_len


from torch import nn, Tensor, functional


class Dense(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs
    ):
        super(Dense, self).__init__(*args, **kwargs)
        self.layer = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        initializer(self.layer.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(inputs)

    def extra_repr(self) -> str:
        return self.layer.extra_repr()


class NoisyDense(nn.Module):
    """See https://arxiv.org/pdf/1706.10295."""

    @staticmethod
    def f(x: Tensor):
        return x.sgn() * (x.abs().sqrt())

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_sigma: float = 0.5,
        use_factorized: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.use_factorized = use_factorized
        self.use_bias = bias

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.eps_w = self.register_buffer(
            "eps_w", torch.empty(out_features, in_features)
        )
        if self.use_bias:
            self.mu_b = nn.Parameter(torch.empty(out_features))
            self.sigma_b = nn.Parameter(torch.empty(out_features))
            self.eps_b = self.register_buffer("eps_b", torch.empty(out_features))
        else:
            self.register_parameter("mu_b", None)
            self.register_parameter("sigma_b", None)
            self.eps_b = self.register_buffer("eps_b", None)

        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self) -> None:
        if self.use_factorized:
            eps_i = torch.randn(1, self.in_features).to(self.mu_w.device)
            eps_j = torch.randn(self.out_features, 1).to(self.mu_w.device)
            self.eps_w = self.f(eps_j) @ self.f(eps_i)
            self.eps_b = self.f(eps_j).reshape(self.out_features)
        else:
            self.eps_w = self.f(torch.randn(self.mu_w.shape)).to(self.mu_w.device)
            if self.use_bias:
                self.eps_b = self.f(torch.randn(size=self.mu_b.shape)).to(
                    self.mu_w.device
                )

    def remove_noise(self) -> None:
        self.eps_w = torch.zeros_like(self.mu_w).to(self.mu_w.device)
        if self.use_bias:
            self.eps_b = torch.zeros_like(self.mu_b).to(self.mu_w.device)

    def reset_parameters(self) -> None:
        p = self.in_features
        if self.use_factorized:
            mu_init = 1.0 / (p**0.5)
            sigma_init = self.initial_sigma / (p**0.5)
        else:
            mu_init = (3.0 / p) ** 0.5
            sigma_init = 0.017

        nn.init.constant_(self.sigma_w, sigma_init)
        nn.init.uniform_(self.mu_w, -mu_init, mu_init)
        if self.use_bias:
            nn.init.constant_(self.sigma_b, sigma_init)
            nn.init.uniform_(self.mu_b, -mu_init, mu_init)

    @property
    def weight(self):
        return self.mu_w + self.sigma_w * self.eps_w

    @property
    def bias(self):
        if self.use_bias:
            return self.mu_b + self.sigma_b * self.eps_b
        else:
            return None

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return functional.F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, initial_sigma={self.initial_sigma}, use_factorized={self.use_factorized}"


def build_dense(in_features: int, out_features: int, sigma: float = 0):
    if sigma == 0:
        return Dense(in_features, out_features)
    else:
        return NoisyDense(in_features, out_features)


class DenseStack(nn.Module):
    def __init__(
        self,
        initial_width: int,
        widths: list[int],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
    ):
        super(DenseStack, self).__init__()
        self.dense_layers: nn.ModuleList = nn.ModuleList()
        self.activation = activation

        assert len(widths) > 0
        self.noisy = noisy_sigma != 0

        current_input_width = initial_width
        for i in range(len(widths)):
            layer = build_dense(
                in_features=current_input_width,
                out_features=widths[i],
                sigma=noisy_sigma,
            )
            self.dense_layers.append(layer)
            current_input_width = widths[i]

        self.initial_width = initial_width
        self._output_len = current_input_width

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        for layer in self.dense_layers:
            layer.initialize(initializer)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.dense_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        assert self.noisy

        for layer in self.dense_layers:
            layer.reset_noise()
        return

    def remove_noise(self) -> None:
        assert self.noisy

        for layer in self.dense_layers:
            layer.remove_noise()
        return

    def extra_repr(self) -> str:
        return f"in_features={self.initial_width}, out_width={self.output_width}, noisy={self.noisy}"

    @property
    def output_width(self):
        return self._output_len


class ResidualStack(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
    ):
        """A sequence of residual layers with the activation function applied after each layer.
        Always applies the minimum zero-padding that ensures the output shape is equal to the input shape.
        Input shape in "BCHW" form, i.e. (batch_size, input_channels, height, width)
        """
        super(ResidualStack, self).__init__()
        self.residual_layers = nn.ModuleList()

        self.activation = activation

        # [B, C_in, H, W]
        assert (
            len(input_shape) == 4
            and len(filters) == len(kernel_sizes) == len(strides)
            and len(filters) > 0
        )

        self.noisy = noisy_sigma != 0
        if self.noisy:
            print("warning: Noisy convolutions not implemented yet")
            # raise NotImplementedError("")

        current_input_channels = input_shape[1]

        for i in range(len(filters)):
            print(current_input_channels)
            layer = Residual(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
            )
            self.residual_layers.append(layer)
            current_input_channels = filters[i]

        self._output_len = current_input_channels

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        def initialize_if_conv(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)

        self.apply(initialize_if_conv)

    def forward(self, inputs):
        x = inputs
        for layer in self.residual_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    def remove_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    @property
    def output_channels(self):
        return self._output_len


class Residual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
    ):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )

        # REGULARIZATION?
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )

        # REGULARIZATION?
        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.relu = nn.ReLU()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        def initialize_if_conv(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)

        self.apply(initialize_if_conv)

    def forward(self, inputs):
        residual = self.downsample(inputs) if self.downsample else inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + residual)
        return x


def kernel_initializer_wrapper(x):
    if x is None:
        return x
    elif isinstance(x, str):
        return prepare_kernel_initializers(x)
    else:
        assert callable(x)
        return x


class RainbowConfig(Config):
    def __init__(self, config_dict: dict, game_config):
        super(RainbowConfig, self).__init__(config_dict, game_config)
        print("RainbowConfig")
        self.residual_layers: list = self.parse_field("residual_layers", [])
        self.conv_layers: list = self.parse_field("conv_layers", [])
        self.dense_layer_widths: int = self.parse_field(
            "dense_layer_widths", [128], tointlists
        )
        self.value_hidden_layer_widths = self.parse_field(
            "value_hidden_layer_widths", [], tointlists
        )
        self.advantage_hidden_layer_widths: int = self.parse_field(
            "advantage_hidden_layer_widths", [], tointlists
        )

        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.5)
        self.eg_epsilon: float = self.parse_field("eg_epsilon", 0.00)
        self.eg_epsilon_final: float = self.parse_field("eg_epsilon_final", 0.00)
        self.eg_epsilon_decay_type: str = self.parse_field(
            "eg_epsilon_decay_type", "linear"
        )
        self.eg_epsilon_final_step: int = self.parse_field(
            "eg_epsilon_final_step", self.training_steps
        )

        self.dueling: bool = self.parse_field("dueling", True)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.soft_update: bool = self.parse_field("soft_update", False)
        self.transfer_interval: int = self.parse_field(
            "transfer_interval", 512, wrapper=int
        )
        self.ema_beta: float = self.parse_field("ema_beta", 0.99)
        self.replay_interval: int = self.parse_field("replay_interval", 1, wrapper=int)
        self.per_alpha: float = self.parse_field("per_alpha", 0.6)
        self.per_beta: float = self.parse_field("per_beta", 0.5)
        self.per_beta_final: float = self.parse_field("per_beta_final", 1.0)
        self.per_epsilon: float = self.parse_field("per_epsilon", 1e-6)
        self.n_step: int = self.parse_field("n_step", 3)
        self.atom_size: int = self.parse_field("atom_size", 51, wrapper=int)
        # assert (
        #     self.atom_size > 1
        # ), "Atom size must be greater than 1, as softmax and Q distribution to Q value calculation requires more than 1 atom"

        # assert not (
        #     self.game.is_image
        #     and len(self.conv_layers) == 0
        #     and len(self.residual_layers) == 0
        # ), "Convolutional layers must be defined for image based games"

        if len(self.conv_layers) > 0:
            assert len(self.conv_layers[0]) == 3

        # maybe don't use a game config, since if tuning for multiple games this should be the same regardless of the game <- (it is really a hyper parameter if you are tuning for multiple games or a game with unknown bounds)

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?) <- might mean this is not in the config but just a part of the model

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

        if self.atom_size != 1:
            assert self.v_min != None and self.v_max != None

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"


class RainbowNetwork(nn.Module):
    def __init__(
        self,
        config: RainbowConfig,
        output_size: int,
        input_shape: Tuple[int],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        self.has_value_hidden_layers = len(config.value_hidden_layer_widths) > 0
        self.has_advantage_hidden_layers = len(config.advantage_hidden_layer_widths) > 0
        if not self.config.dueling:
            assert not (
                self.has_value_hidden_layers or self.has_advantage_hidden_layers
            ), "Value or Advantage hidden layers are only used in dueling networks"

        self.output_size = output_size

        current_shape = input_shape
        B = current_shape[0]

        if self.has_residual_layers:
            assert (
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
            filters, kernel_sizes, strides = to_lists(config.residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=input_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
            filters, kernel_sizes, strides = to_lists(config.conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=input_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert (
                len(current_shape) == 2
            ), "Input shape should be (B, width), got {}".format(current_shape)
            initial_width = current_shape[1]

        if self.config.dueling:
            if self.has_value_hidden_layers:
                # (B, width_in) -> (B, value_in_features) -> (B, atom_size)
                self.value_hidden_layers = DenseStack(
                    initial_width=initial_width,
                    widths=self.config.value_hidden_layer_widths,
                    activation=self.config.activation,
                    noisy_sigma=self.config.noisy_sigma,
                )
                value_in_features = self.value_hidden_layers.output_width
            else:
                value_in_features = initial_width
            # (B, value_in_features) -> (B, atom_size)
            self.value_layer = build_dense(
                in_features=value_in_features,
                out_features=config.atom_size,
                sigma=config.noisy_sigma,
            )

            if self.has_advantage_hidden_layers:
                # (B, width_in) -> (B, advantage_in_features)
                self.advantage_hidden_layers = DenseStack(
                    initial_width=initial_width,
                    widths=self.config.advantage_hidden_layer_widths,
                    activation=self.config.activation,
                    noisy_sigma=self.config.noisy_sigma,
                )
                advantage_in_features = self.advantage_hidden_layers.output_width
            else:
                advantage_in_features = initial_width
            # (B, advantage_in_features) -> (B, output_size * atom_size)
            self.advantage_layer = build_dense(
                in_features=advantage_in_features,
                out_features=output_size * config.atom_size,
                sigma=self.config.noisy_sigma,
            )
        else:
            self.distribution_layer = build_dense(
                in_features=initial_width,
                out_features=self.output_size * self.config.atom_size,
                sigma=self.config.noisy_sigma,
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        if self.has_value_hidden_layers:
            self.value_hidden_layers.initialize(initializer)
        if self.has_advantage_hidden_layers:
            self.advantage_hidden_layers.initialize(initializer)
        if self.config.dueling:
            self.value_layer.initialize(initializer)
            self.advantage_layer.initialize(initializer)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)
        S = S.flatten(1, -1)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_dense_layers:
            S = self.dense_layers(S)

        if self.config.dueling:
            # (B, value_hidden_in) -> (B, value_hidden_out)
            if self.has_value_hidden_layers:
                v = self.value_hidden_layers(S)
            else:
                v = S

            # (B, value_hidden_in || dense_features_out) -> (B, atom_size) -> (B, 1, atom_size)
            v: Tensor = self.value_layer(v).view(-1, 1, self.config.atom_size)

            # (B, adv_hidden_in) -> (B, adv_hidden_out)
            if self.has_advantage_hidden_layers:
                A = self.advantage_hidden_layers(S)
            else:
                A = S

            # (B, adv_hidden_out || dense_features_out) -> (B, output_size * atom_size) -> (B, output_size, atom_size)
            A: Tensor = self.advantage_layer(A).view(
                -1, self.output_size, self.config.atom_size
            )

            # (B, output_size, atom_size) -[mean(1)]-> (B, 1, atom_size)
            a_mean = A.mean(1, keepdim=True)

            # (B, 1, atom_size) +
            # (B, output_size, atom_size) +
            # (B, 1, atom_size)
            # is valid broadcasting operation
            Q = v + A - a_mean

            # -[softmax(2)]-> turns the atom dimension into a valid p.d.f.
            # ONLY CLIP FOR CATEGORICAL CROSS ENTROPY LOSS TO PREVENT NAN
            # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
            # q.clip(1e-3, 1)
        else:
            # (B, dense_features_out) -> (B, output_size, atom_size)
            Q = self.distribution_layer(S).view(
                -1, self.output_size, self.config.atom_size
            )

        if self.config.atom_size == 1:
            return Q.squeeze(-1)
        else:
            return Q.softmax(dim=-1)

    def reset_noise(self):
        if self.config.noisy_sigma != 0:
            if self.has_residual_layers:
                self.residual_layers.reset_noise()
            if self.has_conv_layers:
                self.conv_layers.reset_noise()
            if self.has_dense_layers:
                self.dense_layers.reset_noise()
            if self.has_value_hidden_layers:
                self.value_hidden_layers.reset_noise()
            if self.has_advantage_hidden_layers:
                self.advantage_hidden_layers.reset_noise()
            if self.config.dueling:
                self.value_layer.reset_noise()
                self.advantage_layer.reset_noise()

    def remove_noise(self):
        if self.config.noisy_sigma != 0:
            if self.has_residual_layers:
                self.residual_layers.remove_noise()
            if self.has_conv_layers:
                self.conv_layers.remove_noise()
            if self.has_dense_layers:
                self.dense_layers.remove_noise()
            if self.has_value_hidden_layers:
                self.value_hidden_layers.remove_noise()
            if self.has_advantage_hidden_layers:
                self.advantage_hidden_layers.remove_noise()
            if self.config.dueling:
                self.value_layer.remove_noise()
                self.advantage_layer.remove_noise()


from time import time
import numpy as np
import operator
from typing import Callable


class SegmentTree:
    """Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


class FastSumTree(object):
    # https://medium.com/free-code-camp/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682

    def __init__(self, capacity: int):
        self.capacity = (
            capacity  # number of leaf nodes (final nodes) that contains experiences
        )

        self.tree = np.zeros(2 * self.capacity - 1)  # sub tree
        # self.data = np.zeros(self.capacity, object)  # contains the experiences

    def add(self, idx: int, val: float):
        """Set value in tree."""
        tree_index = idx + self.capacity - 1
        # self.data[self.data_pointer] = data
        self.update(tree_index, val)

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]

    def update(self, tree_index: int, val: float):
        change = val - self.tree[tree_index]
        # print("change", change)
        self.tree[tree_index] = val
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
            # print("new value", self.tree[tree_index])

    def retrieve(self, v: float):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        return leaf_index, self.tree[leaf_index]

    @property
    def total_priority(self):
        return self.tree[0]


class BaseReplayBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int = None,
        compressed_observations: bool = False,
    ):
        self.max_size = max_size
        self.batch_size = batch_size if batch_size is not None else max_size
        self.compressed_observations = compressed_observations

        self.clear()
        assert self.size == 0, "Replay buffer should be empty at initialization"
        assert self.max_size > 0, "Replay buffer should have a maximum size"
        assert self.batch_size > 0, "Replay buffer batch size should be greater than 0"

    def store(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def sample_from_indices(self, indices: list[int]):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def __len__(self):
        return self.size


class Game:
    def __init__(
        self, num_players: int
    ):  # num_actions, discount=1.0, n_step=1, gamma=0.99
        self.length = 0
        self.observation_history = []
        self.rewards = []
        self.policy_history = []
        self.value_history = []
        self.action_history = []
        self.info_history = []

        self.num_players = num_players

    def append(
        self,
        observation,
        reward: int,
        policy,
        value=None,
        action=None,
        info=None,
    ):
        self.observation_history.append(copy.deepcopy(observation))
        self.rewards.append(reward)
        self.policy_history.append(policy)
        self.value_history.append(value)
        self.action_history.append(action)
        self.info_history.append(info)
        self.length += 1

    def set_rewards(self):
        print("Initial Rewards", self.rewards)
        final_reward = self.rewards[-1]
        for i in reversed(range(self.length)):
            self.rewards[i] = (
                final_reward[i % self.num_players]
                # if i % self.num_players == (self.length - 1) % self.num_players
                # else -final_reward
            )
        print("Updated Rewards", self.rewards)

    def __len__(self):
        return self.length


class BaseGameReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
    ):
        super().__init__(max_size=max_size, batch_size=batch_size)

    def store(self, game: Game):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        self.size += 1

    def sample(self):
        move_sum = float(sum([len(game) for game in self.buffer]))
        games: list[Game] = np.random.choice(
            self.buffer,
            self.batch_size,
            p=[len(game) / move_sum for game in self.buffer],
        )

        return [(game, np.random.randint(len(game))) for game in games]

    def clear(self):
        self.buffer: list[Game] = []
        self.size = 0


class BaseDQNReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        print(observation_dtype)
        super().__init__(
            max_size=max_size,
            batch_size=batch_size,
            compressed_observations=compressed_observations,
        )

    def store(
        self,
        observation,
        info: dict,
        action,
        reward: float,
        next_observation,
        next_info: dict,
        done: bool,
        id=None,
    ):
        # compute n-step return and store
        self.id_buffer[self.pointer] = id
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_observation_buffer[self.pointer] = next_observation
        self.done_buffer[self.pointer] = done
        self.info_buffer[self.pointer] = info
        self.next_info_buffer[self.pointer] = next_info

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        if self.compressed_observations:
            self.observation_buffer = np.zeros(self.max_size, dtype=np.object_)
            self.next_observation_buffer = np.zeros(self.max_size, dtype=np.object_)
        else:
            observation_buffer_shape = (self.max_size,) + self.observation_dimensions
            self.observation_buffer = np.zeros(
                observation_buffer_shape, self.observation_dtype
            )
            self.next_observation_buffer = np.zeros(
                observation_buffer_shape, dtype=self.observation_dtype
            )

        self.id_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.action_buffer = np.zeros(self.max_size, dtype=np.uint8)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.done_buffer = np.zeros(self.max_size, dtype=np.bool_)
        self.info_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.next_info_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.pointer = 0
        self.size = 0

    def sample(self):
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
            ids=self.id_buffer[indices],
            info=self.info_buffer[indices],
            next_info=self.next_info_buffer[indices],
        )

    def sample_from_indices(self, indices: list[int]):
        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
            ids=self.id_buffer[indices],
            infos=self.info_buffer[indices],
            next_infos=self.next_info_buffer[indices],
        )

    def __check_id__(self, index: int, id: str) -> bool:
        return self.id_buffer[index] == id


class BasePPOReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: np.dtype,
        max_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        super().__init__(
            max_size=max_size, compressed_observations=compressed_observations
        )

    def store(
        self,
        observation,
        info: dict,
        action,
        value: float,
        log_probability: float,
        reward: float,
        id=None,
    ):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability
        self.info_buffer[self.pointer] = info

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = np.mean(self.advantage_buffer)
        advantage_std = np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (
            advantage_std + 1e-10
        )  # avoid division by zero
        return dict(
            observations=self.observation_buffer,
            actions=self.action_buffer,
            advantages=self.advantage_buffer,
            returns=self.return_buffer,
            log_probabilities=self.log_probability_buffer,
            infos=self.info_buffer,
        )

    def clear(self):
        if self.compressed_observations:
            self.observation_buffer = np.zeros(self.max_size, dtype=np.object_)
            self.next_observation_buffer = np.zeros(self.max_size, dtype=np.object_)
        else:
            observation_buffer_shape = (self.max_size,) + self.observation_dimensions
            self.observation_buffer = np.zeros(
                observation_buffer_shape, self.observation_dtype
            )
            self.next_observation_buffer = np.zeros(
                observation_buffer_shape, dtype=self.observation_dtype
            )
        self.action_buffer = np.zeros(self.max_size, dtype=np.int8)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.advantage_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.return_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.value_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.log_probability_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.info_buffer = np.zeros(self.max_size, dtype=np.object_)

        self.pointer = 0
        self.trajectory_start_index = 0
        self.size = 0

    def finish_trajectory(self, last_value: float = 0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.gae_lambda
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        # print(discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda))
        # print(discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)[:-1])
        # print(self.advantage_buffer)

        self.trajectory_start_index = self.pointer


class NStepReplayBuffer(BaseDQNReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
        compressed_observations: bool = False,
        num_players: int = 1,
    ):
        self.n_step = n_step
        self.gamma = gamma
        self.num_players = num_players
        super().__init__(
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            max_size=max_size,
            batch_size=batch_size,
            compressed_observations=compressed_observations,
        )

    def store(
        self,
        observation,
        info: dict,
        action,
        reward: float,
        next_observation,
        next_info: dict,
        done: bool,
        id=None,
        player: int = 0,
    ):
        """Store a (s_t, a, r, s_t+1) transtion to the replay buffer.
           Returns a valid generated n-step transition (s_t-n, a, r, s_t) with the
           inputted observation as the next_observation (s_t)

        Returns:
            (s_t-n, a, r, s_t): where r is the n-step return calculated with the replay buffer's gamma
        """
        transition = (
            observation,
            info,
            action,
            reward,
            next_observation,
            next_info,
            done,
        )
        # print("store t:", transition)
        self.n_step_buffers[player].append(transition)
        if len(self.n_step_buffers[player]) < self.n_step:
            return None

        # compute n-step return and store
        reward, next_observation, next_info, done = self._get_n_step_info(player)
        observation, info, action = self.n_step_buffers[player][0][:3]
        n_step_transition = (
            observation,
            info,
            action,
            reward,
            next_observation,
            next_info,
            done,
        )
        super().store(*n_step_transition, id=id)
        return n_step_transition

    def clear(self):
        super().clear()
        self.n_step_buffers = [
            deque(maxlen=self.n_step) for q in range(self.num_players)
        ]

    def _get_n_step_info(self, player: int = 0):
        reward, next_observation, next_info, done = self.n_step_buffers[player][-1][-4:]

        for transition in reversed(list(self.n_step_buffers[player])[:-1]):
            r, n_o, n_i, d = transition[-4:]
            reward = r + self.gamma * reward * (1 - d)
            next_observation, next_info, done = (
                (n_o, n_i, d) if d else (next_observation, next_info, done)
            )

        return reward, next_observation, next_info, done


class PrioritizedNStepReplayBuffer(NStepReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # epsilon=0.01,
        n_step: float = 1,
        gamma: float = 0.99,
        compressed_observations: bool = False,
        num_players: int = 1,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma > 0 and gamma <= 1

        self.initial_max_priority = max_priority
        super(PrioritizedNStepReplayBuffer, self).__init__(
            observation_dimensions,
            observation_dtype,
            max_size,
            batch_size,
            n_step=n_step,
            gamma=gamma,
            compressed_observations=compressed_observations,
            num_players=num_players,
        )

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.beta = beta
        # self.epsilon = epsilon

    def store(
        self,
        observation,
        info: dict,
        action,
        reward: float,
        next_observation,
        next_info: dict,
        done: bool,
        id=None,
        priority: float = None,
        player: int = 0,
    ):
        transition = super().store(
            observation,
            info,
            action,
            reward,
            next_observation,
            next_info,
            done,
            id,
            player=player,
        )

        if priority is None:
            priority = self.max_priority**self.alpha
            self.max_priority = max(
                self.max_priority, priority
            )  # could remove and clip priorities in experience replay isntead

        if transition:
            self.sum_tree[self.tree_pointer] = priority**self.alpha
            self.min_tree[self.tree_pointer] = priority**self.alpha
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        return transition

    def set_beta(self, beta: float):
        self.beta = beta

    def store_batch(self, batch):
        (
            observations,
            infos,
            actions,
            rewards,
            next_observations,
            next_infos,
            dones,
            ids,
            priorities,
        ) = batch
        for i in range(len(observations)):
            self.store(
                observations[i],
                infos[i],
                actions[i],
                rewards[i],
                next_observations[i],
                next_infos[i],
                dones[i],
                ids[i],
                priorities[i],
            )

    def sample(self, throw_exception=True) -> dict:
        if len(self) < self.batch_size:
            if throw_exception:
                raise "Only {} elements in buffer expected at least {}".format(
                    len(self), self.batch_size
                )
            else:
                return None

        if self.alpha != 0.0:
            indices = self._sample_proportional()
        else:
            indices = np.random.choice(self.size, size=self.batch_size, replace=False)
            # print(indices)
        weights = np.array([self._calculate_weight(i) for i in indices])

        n_step_samples = self.sample_from_indices(indices)
        # print(n_step_samples)
        n_step_samples.update(dict(weights=weights, indices=indices))
        # print(n_step_samples)

        return n_step_samples

    def clear(self):
        super().clear()
        self.max_priority = self.initial_max_priority  # (initial) priority
        self.tree_pointer = 0

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def update_priorities(self, indices: list[int], priorities: list[float], ids=None):
        # necessary for shared replay buffer
        if ids is not None:
            assert len(priorities) == len(ids) == len(indices)
            assert priorities.shape == ids.shape == indices.shape

            for index, id, priority in zip(indices, ids, priorities):
                assert (
                    priority > 0
                ), "Negative priority: {} \n All priorities {}".format(
                    priority, priorities
                )
                assert 0 <= index < len(self)

                if self.id_buffer[index] != id:
                    continue

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(self.max_priority, priority)
        else:
            assert len(indices) == len(priorities)
            for index, priority in zip(indices, priorities):
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(
                    self.max_priority, priority
                )  # could remove and clip priorities in experience replay isntead

        return priorities**self.alpha

    def _sample_proportional(self):
        indices = []
        total_priority = self.sum_tree.sum(0, len(self) - 1)
        priority_segment = total_priority / self.batch_size

        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            index = self.sum_tree.retrieve(upperbound)
            indices.append(index)

        return indices

    def _calculate_weight(self, index: int):
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * len(self)) ** (-self.beta)
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-self.beta)
        weight = weight / max_weight

        return weight


class FastPrioritizedReplayBuffer(NStepReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size: int = 32,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # epsilon=0.01,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma > 0 and gamma <= 1

        super(FastPrioritizedReplayBuffer, self).__init__(
            observation_dimensions, max_size, batch_size, n_step=n_step, gamma=gamma
        )

        self.max_priority = max_priority  # (initial) priority
        self.min_priority = max_priority
        self.tree_pointer = 0

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.beta = beta
        # self.epsilon = epsilon

        self.tree = FastSumTree(self.max_size)

    def store(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
    ):
        transition = super().store(observation, action, reward, next_observation, done)

        # max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        # if max_priority == 0:
        #     max_priority = self.max_priority

        if transition:
            self.tree.add(self.tree_pointer, self.max_priority)
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        return transition

    def sample(self):
        assert len(self) >= self.batch_size

        priority_segment = self.tree.total_priority / self.batch_size
        indices, weights = np.empty((self.batch_size,), dtype=np.int32), np.empty(
            (self.batch_size, 1), dtype=np.float32
        )
        for i in range(self.batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority = self.tree.retrieve(value)
            sampling_probabilities = priority / self.tree.total_priority
            # weights[i, 0] = (self.batch_size * sampling_probabilities) ** -beta
            weights[i, 0] = (len(self) * sampling_probabilities) ** -self.beta
            indices[i] = index - self.tree.capacity + 1
            indices[i] = index - self.tree.capacity + 1

        # max_weight = max(weights)
        max_weight = (
            len(self) * self.min_priority / self.tree.total_priority
        ) ** -self.beta
        weights = weights / max_weight

        # print(weights)
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree Time ", time() - time1)
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays")
        # time2 = 0
        # time2 = time()
        observations = self.observation_buffer[indices]
        next_observations = self.next_observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.done_buffer[indices]
        # weights = np.array([self._calculate_weight(i, beta) for i in indices])
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays Time ", time() - time2)

        # print("Sampling from PrioritizedReplayBuffer Time ", time() - time1)
        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: list[int], priorities: list[float]):
        assert len(indices) == len(priorities)
        # priorities += self.epsilon

        for index, priority in zip(indices, priorities):
            assert priority > 0, "Negative priority: {}".format(priority)
            # assert 0 <= index < len(self)
            # self.tree[index] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority**self.alpha)
            self.min_priority = min(self.min_priority, priority**self.alpha)
            # priority = np.clip(priority, self.epsilon, self.max_priority)
            self.tree.update(index + self.tree.capacity - 1, priority**self.alpha)


class RainbowAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: RainbowConfig,
        name=f"rainbow_{current_timestamp():.1f}",
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
        from_checkpoint=False,
    ):
        super(RainbowAgent, self).__init__(env, config, name, device=device)
        self.model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
        )
        self.target_model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
        )

        if not self.config.kernel_initializer == None:
            self.model.initialize(self.config.kernel_initializer)

        self.model.to(device)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        self.replay_buffer = PrioritizedNStepReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            max_priority=1.0,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            # epsilon=config["per_epsilon"],
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
            compressed_observations=(
                self.env.lz4_compress if hasattr(self.env, "lz4_compress") else False
            ),
            num_players=self.config.game.num_players,
        )

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?)
        # self.v_min = self.config.v_min
        # self.v_max = self.config.v_max

        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=device,
        ).to(device)
        """row vector Tensor(atom_size)
        """

        self.eg_epsilon = self.config.eg_epsilon

        self.stats = {
            "score": [],
            "loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

    def checkpoint_model_weights(self, checkpoint):
        checkpoint = super().checkpoint_model_weights(checkpoint)
        checkpoint["target_model"] = self.target_model.state_dict()

    def load_model_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.target_model.eval()

    def predict(self, states, *args, **kwargs) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.model(state_input)
        return q_distribution

    def predict_target(self, states) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.target_model(state_input)
        return q_distribution

    def select_actions(
        self, distribution, info: dict = None, mask_actions: bool = True
    ):
        assert info is not None if mask_actions else True, "Need info to mask actions"
        # print(info)
        if self.config.atom_size > 1:
            q_values = distribution * self.support
            q_values = q_values.sum(2, keepdim=False)
        else:
            q_values = distribution
        if mask_actions:
            legal_moves = get_legal_moves(info)
            q_values = action_mask(
                q_values, legal_moves, mask_value=-float("inf"), device=self.device
            )
        # print("Q Values", q_values)
        # q_values with argmax ties
        # selected_actions = torch.stack(
        #     [
        #         torch.tensor(np.random.choice(np.where(x.cpu() == x.cpu().max())[0]))
        #         for x in q_values
        #     ]
        # )
        # print(selected_actions)
        selected_actions = q_values.argmax(1, keepdim=False)
        return selected_actions

    def learn(self) -> np.ndarray:
        losses = np.zeros(self.config.training_iterations)
        for i in range(self.config.training_iterations):
            samples = self.replay_buffer.sample()
            loss = self.learn_from_sample(samples)
            losses[i] = loss
        return losses

    def learn_from_sample(self, samples: dict):
        observations, weights, actions = (
            samples["observations"],
            samples["weights"],
            torch.from_numpy(samples["actions"]).to(self.device).long(),
        )
        # print("actions", actions)

        # print("Observations", observations)
        # (B, outputs, atom_size) -[index action dimension by actions]> (B, atom_size)
        online_predictions = self.predict(observations)[
            range(self.config.minibatch_size), actions
        ]
        # for param in self.model.parameters():
        #     print(param)
        # print(self.predict(observations))
        # print(online_predictions)
        # (B, atom_size)
        if self.config.atom_size > 1:
            assert isinstance(
                self.config.loss_function, KLDivergenceLoss
            ) or isinstance(
                self.config.loss_function, CategoricalCrossentropyLoss
            ), "Only KLDivergenceLoss and CategoricalCrossentropyLoss are supported for atom_size > 1, recieved {}".format(
                self.config.loss_function
            )
            target_predictions = self.compute_target_distributions(samples)
        else:
            # print("using default dqn loss")
            assert isinstance(self.config.loss_function, HuberLoss) or isinstance(
                self.config.loss_function, MSELoss
            ), "Only HuberLoss or MSELoss are supported for atom_size = 1, recieved {}".format(
                self.config.loss_function
            )
            next_observations, rewards, dones = (
                torch.from_numpy(samples["next_observations"]).to(self.device),
                torch.from_numpy(samples["rewards"]).to(self.device),
                torch.from_numpy(samples["dones"]).to(self.device),
            )
            next_infos = samples["next_infos"]
            target_predictions = self.predict_target(next_observations)  # next q values
            # print("Next q values", target_predictions)
            # print("Current q values", online_predictions)
            # print(self.predict(next_observations))
            next_actions = self.select_actions(
                self.predict(next_observations),  # current q values
                info=next_infos,
                mask_actions=self.config.game.has_legal_moves,
            )
            # print("Next actions", next_actions)
            target_predictions = target_predictions[
                range(self.config.minibatch_size), next_actions
            ]  # this might not work
            # print(target_predictions)
            target_predictions = (
                rewards + self.config.discount_factor * (~dones) * target_predictions
            )
            # print(target_predictions)

        # print("predicted", online_distributions)
        # print("target", target_distributions)

        weights_cuda = torch.from_numpy(weights).to(torch.float32).to(self.device)
        # (B)
        elementwise_loss = self.config.loss_function(
            online_predictions, target_predictions
        )
        # print("Loss", elementwise_loss.mean())
        assert torch.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )
        assert (
            elementwise_loss.shape == weights_cuda.shape
        ), "Loss Shape: {}, Weights Shape: {}".format(
            elementwise_loss.shape, weights_cuda.shape
        )
        loss = elementwise_loss * weights_cuda
        self.optimizer.zero_grad()
        loss.mean().backward()
        if self.config.clipnorm > 0:
            # print("clipnorm", self.config.clipnorm)
            clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

        self.optimizer.step()
        self.update_replay_priorities(
            samples=samples,
            priorities=elementwise_loss.detach().to("cpu").numpy()
            + self.config.per_epsilon,
        )
        self.model.reset_noise()
        self.target_model.reset_noise()
        return loss.detach().to("cpu").mean().item()

    def update_replay_priorities(self, samples, priorities):
        self.replay_buffer.update_priorities(samples["indices"], priorities)

    def compute_target_distributions(self, samples):
        # print("computing target distributions")
        with torch.no_grad():
            discount_factor = self.config.discount_factor**self.config.n_step
            delta_z = (self.config.v_max - self.config.v_min) / (
                self.config.atom_size - 1
            )
            next_observations, rewards, dones = (
                samples["next_observations"],
                torch.from_numpy(samples["rewards"]).to(self.device).view(-1, 1),
                torch.from_numpy(samples["dones"]).to(self.device).view(-1, 1),
            )
            online_distributions = self.predict(next_observations)
            target_distributions = self.predict_target(next_observations)

            # print(samples["next_infos"])
            next_actions = self.select_actions(
                online_distributions,
                info=samples["next_infos"],
                mask_actions=self.config.game.has_legal_moves,
            )  # {} is the info but we are not doing action masking yet
            # (B, outputs, atom_size) -[index by [0..B-1, a_0..a_B-1]]> (B, atom_size)
            probabilities = target_distributions[
                range(self.config.minibatch_size), next_actions
            ]
            # print(probabilities)

            # (B, 1) + k(B, atom_size) * (B, atom_size) -> (B, atom_size)
            Tz = (rewards + discount_factor * (~dones) * self.support).clamp(
                self.config.v_min, self.config.v_max
            )
            # print("Tz", Tz)

            # all elementwise
            b: torch.Tensor = (Tz - self.config.v_min) / delta_z
            l, u = (
                torch.clamp(b.floor().long(), 0, self.config.atom_size - 1),
                torch.clamp(b.ceil().long(), 0, self.config.atom_size - 1),
            )
            # print("b", b)
            # print("l", l)
            # print("u", u)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.atom_size - 1)) * (l == u)] += 1
            # print("fixed l", l)
            # print("fixed u", u)
            # dones = dones.squeeze()
            # masked_probs = torch.ones_like(probabilities) / self.config.atom_size
            # masked_probs[~dones] = probabilities[~dones]

            m = torch.zeros_like(probabilities)
            m.scatter_add_(dim=1, index=l, src=probabilities * ((u.float()) - b))
            m.scatter_add_(dim=1, index=u, src=probabilities * ((b - l.float())))
            # print("old_m", (m * self.support).sum(-1))

            # projected_distribution = torch.zeros_like(probabilities)
            # projected_distribution.scatter_add_(
            #     dim=1, index=l, src=masked_probs * (u.float() - b)
            # )
            # projected_distribution.scatter_add_(
            #     dim=1, index=u, src=masked_probs * (b - l.float())
            # )
            # print("m", (projected_distribution * self.support).sum(-1))
            return m

    def fill_replay_buffer(self):
        print("replay buffer size:", self.replay_buffer.size)
        with torch.no_grad():
            state, info = self.env.reset()
            target_size = self.config.min_replay_buffer_size
            while self.replay_buffer.size < target_size:
                if (self.replay_buffer.size % (target_size // 100)) == 0:
                    print(
                        f"filling replay buffer: {self.replay_buffer.size} / ({target_size})"
                    )
                # dist = self.predict(state)
                # action = self.select_actions(dist).item()
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action
                )
                done = terminated or truncated
                # print(state)
                self.replay_buffer.store(
                    state, info, action, reward, next_state, next_info, done
                )
                # print(self.replay_buffer.observation_buffer[0])
                state = next_state
                info = next_info
                if done:
                    state, info = self.env.reset()
                # gc.collect()

    def update_target_model(self):
        if self.config.soft_update:
            for wt, wp in zip(self.target_model.parameters(), self.model.parameters()):
                wt.copy_(self.config.ema_beta * wt + (1 - self.config.ema_beta) * wp)
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_eg_epsilon(self, training_step):
        if self.config.eg_epsilon_decay_type == "linear":
            # print("decaying eg epsilon linearly")
            self.eg_epsilon = update_linear_schedule(
                self.config.eg_epsilon_final,
                self.config.eg_epsilon_final_step,
                self.config.eg_epsilon,
                training_step,
            )
        elif self.config.eg_epsilon_decay_type == "inverse_sqrt":
            self.eg_epsilon = update_inverse_sqrt_schedule(
                self.config.eg_epsilon,
                training_step,
            )
        else:
            raise ValueError(
                "Invalid epsilon decay type: {}".format(
                    self.config.eg_epsilon_decay_type
                )
            )

    def train(self):
        super().train()
        start_time = time() - self.training_time
        score = 0
        target_model_updated = (False, False)  # (score, loss)
        self.fill_replay_buffer()

        state, info = self.env.reset()

        while self.training_step < self.config.training_steps:


            with torch.no_grad():
                for _ in range(self.config.replay_interval):
                    values = self.predict(state)
                    # print(values)
                    action = epsilon_greedy_policy(
                        values,
                        info,
                        self.eg_epsilon,
                        wrapper=lambda values, info: self.select_actions(
                            values, info
                        ).item(),
                    )
                    # print("Action", action)
                    # print("Epislon Greedy Epsilon", self.eg_epsilon)
                    next_state, reward, terminated, truncated, next_info = (
                        self.env.step(action)
                    )
                    done = terminated or truncated
                    # print("State", state)
                    self.replay_buffer.store(
                        state, info, action, reward, next_state, next_info, done
                    )
                    state = next_state
                    info = next_info
                    score += reward
                    self.replay_buffer.set_beta(
                        update_per_beta(
                            self.replay_buffer.beta,
                            self.config.per_beta_final,
                            self.training_steps,
                            self.config.per_beta,
                        )
                    )

                    if done:
                        state, info = self.env.reset()
                        score_dict = {
                            "score": score,
                            "target_model_updated": target_model_updated[0],
                        }
                        self.stats["score"].append(score_dict)
                        target_model_updated = (False, target_model_updated[1])
                        score = 0

            self.update_eg_epsilon(self.training_step + 1)
            # print("replay buffer size", len(self.replay_buffer))
            for minibatch in range(self.config.num_minibatches):
                if len(self.replay_buffer) < self.config.min_replay_buffer_size:
                    break
                losses = self.learn()
                # print(losses)
                loss_mean = losses.mean()
                # could do things other than taking the mean here
                self.stats["loss"].append(
                    {"loss": loss_mean, "target_model_updated": target_model_updated[1]}
                )
                target_model_updated = (target_model_updated[0], False)

            if self.training_step % self.config.transfer_interval == 0:
                target_model_updated = (True, True)
                # stats["test_score"].append(
                #     {"target_model_weight_update": training_step}
                # )
                self.update_target_model()

            if self.training_step % self.checkpoint_interval == 0:
                # print(self.stats["score"])
                # print(len(self.replay_buffer))
                self.training_time = time() - start_time
                self.total_environment_steps = (
                    self.training_step * self.config.replay_interval
                )
                self.save_checkpoint()
            # gc.collect()
            self.training_step += 1

        self.training_time = time() - start_time
        self.total_environment_steps = self.training_step * self.config.replay_interval
        self.save_checkpoint()
        self.env.close()


from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np

config_dict = {
    "conv_layers": [
        (32, 8, 4),
        (64, 4, 2),
        (64, 3, 1),
    ],
    "dense_layers_widths": [512],
    "value_hidden_layers_widths": [],  #
    "advatage_hidden_layers_widths": [],  #
    "adam_epsilon": 1.5e-4,
    "learning_rate": 0.00025 / 4,
    "training_steps": 50000000,  # Agent saw 200,000,000 frames
    "per_epsilon": 1e-6,  #
    "per_alpha": 0.5,
    "per_beta": 0.4,
    "minibatch_size": 32,
    "replay_buffer_size": 1000000,
    "min_replay_buffer_size": 80000,  # 80000
    "transfer_interval": 32000,
    "n_step": 3,
    "kernel_initializer": "orthogonal",  #
    "loss_function": KLDivergenceLoss(),
    "clipnorm": 0.0,  #
    "discount_factor": 0.99,
    "atom_size": 51,
    "replay_interval": 4,
}


game_config = AtariConfig()
config = RainbowConfig(config_dict, game_config)


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


env = gym.make(
    "MsPacmanNoFrameskip-v4", render_mode="rgb_array", max_episode_steps=108000
)
env = AtariPreprocessing(env, terminal_on_life_loss=True)
env = FrameStack(env, 4, lz4_compress=True)
agent = RainbowAgent(env, config, name="Rainbow_Atari_MsPacmanNoFrameskip-v4")
agent.checkpoint_interval = 1000
agent.train()
