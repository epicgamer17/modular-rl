#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "min_max_stats.hpp"
#include "nodes.hpp"

namespace rainbow::search {

enum class ScoringMethodType {
    kUcb = 0,
    kGumbel = 1,
    kLeastVisited = 2,
    kPrior = 3,
    kQValue = 4,
};

struct ScoringConfig {
    double pb_c_init = 1.25;
    double pb_c_base = 19652.0;
    double unvisited_value_bootstrap = 0.0;
};

using ScoreFunction = std::vector<double> (*)(
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

double score_initial(ScoringMethodType type, double prior, int action);

std::vector<double> compute_ucb_scores(
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

std::vector<double> compute_gumbel_scores(
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

std::vector<double> compute_gumbel_scores_with_policy(
    const NodeArena& arena,
    int node_index,
    const std::vector<double>& improved_policy,
    const ScoringConfig& config);

std::vector<double> compute_least_visited_scores(
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

std::vector<double> compute_prior_scores(
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

std::vector<double> compute_q_value_scores(
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

ScoreFunction resolve_scoring_function(ScoringMethodType type);

std::vector<double> compute_scores(
    ScoringMethodType type,
    const NodeArena& arena,
    int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config);

}  // namespace rainbow::search
