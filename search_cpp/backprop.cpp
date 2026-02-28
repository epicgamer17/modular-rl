#include "backprop.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace rainbow::search {

namespace {

int normalize_player(const int player, const int num_players) {
    if (num_players <= 0) {
        return 0;
    }
    if (player < 0) {
        return 0;
    }
    return player % num_players;
}

double node_value_for_parent(const NodeArena& arena, const int node_index) {
    const Node& node = arena.node(node_index);
    if (node.visits() > 0) {
        return node.value();
    }
    if (node.is_decision()) {
        return arena.decision(node_index).network_value();
    }
    return arena.chance(node_index).network_value();
}

void ensure_parent_child_slot(Node& parent, const int action) {
    if (action < 0) {
        throw std::invalid_argument("Action index must be non-negative.");
    }
    const std::size_t idx = static_cast<std::size_t>(action);
    if (parent.mutable_child_visits().size() <= idx) {
        parent.mutable_child_visits().resize(idx + 1, 0.0);
    }
    if (parent.mutable_child_values().size() <= idx) {
        parent.mutable_child_values().resize(idx + 1, 0.0);
    }
    if (parent.mutable_child_priors().size() <= idx) {
        parent.mutable_child_priors().resize(idx + 1, 0.0);
    }
}

double expected_chance_value(const ChanceNode& node) {
    const std::vector<double>& probs = node.child_priors();
    const std::vector<double>& vals = node.child_values();
    if (probs.empty()) {
        return 0.0;
    }
    const std::size_t n = std::min(probs.size(), vals.size());
    if (n == 0) {
        return 0.0;
    }

    double prob_sum = 0.0;
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double p = probs[i];
        prob_sum += p;
        acc += p * vals[i];
    }
    if (prob_sum <= 0.0) {
        return 0.0;
    }
    return acc / prob_sum;
}

double reduce_decision_children(
    const DecisionNode& node,
    const bool take_min) {
    const std::vector<double>& child_visits = node.child_visits();
    const std::vector<double>& child_values = node.child_values();
    const std::size_t n = std::min(child_visits.size(), child_values.size());
    if (n == 0) {
        return node.value(node.network_value());
    }

    bool any_visited = false;
    double best = take_min ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (child_visits[i] <= 0.0) {
            continue;
        }
        any_visited = true;
        if (take_min) {
            best = std::min(best, child_values[i]);
        } else {
            best = std::max(best, child_values[i]);
        }
    }
    if (!any_visited) {
        return node.value(node.network_value());
    }
    return best;
}

void update_parent_child_stats_with_mean(
    Node& parent,
    const int action,
    const double target_q) {
    ensure_parent_child_slot(parent, action);
    const std::size_t idx = static_cast<std::size_t>(action);
    std::vector<double>& visits = parent.mutable_child_visits();
    std::vector<double>& values = parent.mutable_child_values();
    visits[idx] += 1.0;
    const double n = visits[idx];
    values[idx] += (target_q - values[idx]) / n;
}

void update_parent_child_stats_direct(
    Node& parent,
    const int action,
    const double target_q) {
    ensure_parent_child_slot(parent, action);
    const std::size_t idx = static_cast<std::size_t>(action);
    parent.mutable_child_visits()[idx] += 1.0;
    parent.mutable_child_values()[idx] = target_q;
}

void average_discounted_return_backprop(
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    const double leaf_value,
    const int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config) {
    const int n = static_cast<int>(search_path.size());
    if (n == 0) {
        return;
    }

    const int num_players = std::max(1, config.num_players);
    std::vector<double> acc(static_cast<std::size_t>(num_players), 0.0);
    for (int p = 0; p < num_players; ++p) {
        acc[static_cast<std::size_t>(p)] = (p == normalize_player(leaf_to_play, num_players)) ? leaf_value : -leaf_value;
    }

    for (int i = n - 1; i >= 0; --i) {
        Node& node = arena.node(search_path[static_cast<std::size_t>(i)]);
        const int node_player = normalize_player(node.to_play(), num_players);
        const double total = acc[static_cast<std::size_t>(node_player)];
        node.add_value(total);
        node.add_visit(1);

        if (i > 0) {
            const int parent_idx = search_path[static_cast<std::size_t>(i - 1)];
            const int child_idx = search_path[static_cast<std::size_t>(i)];
            const int action = action_path[static_cast<std::size_t>(i - 1)];
            Node& parent = arena.node(parent_idx);
            Node& child = arena.node(child_idx);

            const int acting_player = normalize_player(parent.to_play(), num_players);
            if (child.is_decision()) {
                const double reward = arena.decision(child_idx).reward();
                for (int p = 0; p < num_players; ++p) {
                    const double sign = (p == acting_player) ? 1.0 : -1.0;
                    acc[static_cast<std::size_t>(p)] = sign * reward + config.discount_factor * acc[static_cast<std::size_t>(p)];
                }
            }

            const double child_q = compute_child_q_from_parent(arena, parent_idx, child_idx, config);
            min_max_stats.update(child_q);

            const int parent_player = normalize_player(parent.to_play(), num_players);
            const double target_q = acc[static_cast<std::size_t>(parent_player)];
            update_parent_child_stats_with_mean(parent, action, target_q);
        } else {
            min_max_stats.update(node.value());
        }
    }
}

void minimax_backprop(
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    const double leaf_value,
    const int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config) {
    const int n = static_cast<int>(search_path.size());
    if (n == 0) {
        return;
    }

    const int leaf_idx = search_path.back();
    Node& leaf = arena.node(leaf_idx);
    leaf.add_visit(1);
    const double signed_leaf_value = (leaf.to_play() == leaf_to_play) ? leaf_value : -leaf_value;
    leaf.set_value_sum(signed_leaf_value * static_cast<double>(leaf.visits()));
    min_max_stats.update(signed_leaf_value);

    for (int i = n - 2; i >= 0; --i) {
        const int node_idx = search_path[static_cast<std::size_t>(i)];
        const int child_idx = search_path[static_cast<std::size_t>(i + 1)];
        const int action = action_path[static_cast<std::size_t>(i)];
        Node& node = arena.node(node_idx);
        node.add_visit(1);

        const double q_val = compute_child_q_from_parent(arena, node_idx, child_idx, config);
        update_parent_child_stats_direct(node, action, q_val);

        if (node.is_decision()) {
            const DecisionNode& decision = arena.decision(node_idx);
            bool take_min = false;
            if (config.alternating_minimax && config.perspective_player >= 0) {
                take_min = (decision.to_play() != config.perspective_player);
            }
            const double best = reduce_decision_children(decision, take_min);
            node.set_value_sum(best * static_cast<double>(node.visits()));
        } else {
            const ChanceNode& chance = arena.chance(node_idx);
            const double expected = expected_chance_value(chance);
            node.set_value_sum(expected * static_cast<double>(node.visits()));
        }

        min_max_stats.update(node.value());
    }
}

}  // namespace

void AverageDiscountedReturnBackpropagator::backpropagate(
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    const double leaf_value,
    const int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config) const {
    average_discounted_return_backprop(
        arena,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config);
}

void MinimaxBackpropagator::backpropagate(
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    const double leaf_value,
    const int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config) const {
    minimax_backprop(
        arena,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config);
}

BackpropFunction resolve_backprop_function(const BackpropMethodType type) {
    switch (type) {
        case BackpropMethodType::kAverageDiscountedReturn:
            return &average_discounted_return_backprop;
        case BackpropMethodType::kMinimax:
            return &minimax_backprop;
    }
    throw std::invalid_argument("Unsupported backprop method.");
}

void backpropagate_with_method(
    const BackpropMethodType type,
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    const double leaf_value,
    const int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config) {
    const BackpropFunction fn = resolve_backprop_function(type);
    fn(
        arena,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config);
}

double compute_child_q_from_parent(
    const NodeArena& arena,
    const int parent_index,
    const int child_index,
    const BackpropConfig& config) {
    const Node& parent = arena.node(parent_index);
    const Node& child = arena.node(child_index);

    if (child.is_chance()) {
        return node_value_for_parent(arena, child_index);
    }

    const DecisionNode& decision_child = arena.decision(child_index);
    const double reward = decision_child.reward();
    const double child_value = node_value_for_parent(arena, child_index);
    const double sign = (decision_child.to_play() == parent.to_play()) ? 1.0 : -1.0;
    return reward + config.discount_factor * (sign * child_value);
}

}  // namespace rainbow::search
