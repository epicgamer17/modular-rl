#pragma once

#include <vector>

#include "min_max_stats.hpp"
#include "nodes.hpp"

namespace rainbow::search {

enum class BackpropMethodType {
    kAverageDiscountedReturn = 0,
    kMinimax = 1,
};

struct BackpropConfig {
    int num_players = 2;
    double discount_factor = 1.0;
    bool alternating_minimax = false;
    int perspective_player = -1;
};

class Backpropagator {
public:
    virtual ~Backpropagator() = default;

    virtual void backpropagate(
        NodeArena& arena,
        const std::vector<int>& search_path,
        const std::vector<int>& action_path,
        double leaf_value,
        int leaf_to_play,
        MinMaxStats& min_max_stats,
        const BackpropConfig& config) const = 0;
};

class AverageDiscountedReturnBackpropagator final : public Backpropagator {
public:
    void backpropagate(
        NodeArena& arena,
        const std::vector<int>& search_path,
        const std::vector<int>& action_path,
        double leaf_value,
        int leaf_to_play,
        MinMaxStats& min_max_stats,
        const BackpropConfig& config) const override;
};

class MinimaxBackpropagator final : public Backpropagator {
public:
    void backpropagate(
        NodeArena& arena,
        const std::vector<int>& search_path,
        const std::vector<int>& action_path,
        double leaf_value,
        int leaf_to_play,
        MinMaxStats& min_max_stats,
        const BackpropConfig& config) const override;
};

using BackpropFunction = void (*)(
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    double leaf_value,
    int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config);

BackpropFunction resolve_backprop_function(BackpropMethodType type);

void backpropagate_with_method(
    BackpropMethodType type,
    NodeArena& arena,
    const std::vector<int>& search_path,
    const std::vector<int>& action_path,
    double leaf_value,
    int leaf_to_play,
    MinMaxStats& min_max_stats,
    const BackpropConfig& config);

double compute_child_q_from_parent(
    const NodeArena& arena,
    int parent_index,
    int child_index,
    const BackpropConfig& config);

}  // namespace rainbow::search
