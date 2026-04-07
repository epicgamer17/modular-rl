#include "scoring.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace search {

namespace {

std::size_t child_count(const Node& node) {
    return node.child_priors().size();
}

void copy_or_pad(
    const std::vector<double>& source,
    const std::size_t n,
    std::vector<double>& out) {
    out.assign(n, 0.0);
    const std::size_t k = std::min(n, source.size());
    for (std::size_t i = 0; i < k; ++i) {
        out[i] = source[i];
    }
}

void compute_ucb_scores_into(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config,
    std::vector<double>& scores) {
    const Node& node = arena.node(node_index);
    const std::size_t n = child_count(node);
    if (n == 0) {
        scores.clear();
        return;
    }

    scores.assign(n, 0.0);
    const std::vector<double>& priors = node.child_priors();
    const std::vector<double>& visits = node.child_visits();
    const std::vector<double>& values = node.child_values();
    assert(priors.size() == n);
    assert(visits.size() == n);
    assert(values.size() == n);

    const double pb_c_base = std::max(config.pb_c_base, 1e-12);
    double pb_c = std::log(
        (static_cast<double>(node.visits()) + pb_c_base + 1.0) / pb_c_base) +
        config.pb_c_init;
    pb_c *= std::sqrt(std::max(0.0, static_cast<double>(node.visits())));

    for (std::size_t i = 0; i < n; ++i) {
        const double child_visit = visits[i];
        const double prior = priors[i];
        const double q = (child_visit > 0.0)
            ? values[i]
            : node.value();
        const double prior_score = (pb_c / (child_visit + 1.0)) * prior;
        const double value_score = min_max_stats.normalize(q);
        scores[i] = prior_score + value_score;
    }
}

void compute_gumbel_scores_with_policy_into(
    const NodeArena& arena,
    const int node_index,
    const std::vector<double>& improved_policy,
    std::vector<double>& scores) {
    const Node& node = arena.node(node_index);
    const std::size_t n = child_count(node);
    if (n == 0) {
        scores.clear();
        return;
    }

    const std::vector<double>& visits = node.child_visits();
    const std::vector<double>& source =
        improved_policy.empty() ? node.child_priors() : improved_policy;
    copy_or_pad(source, n, scores);
    const double sum_n = std::accumulate(visits.begin(), visits.end(), 0.0);
    const double denom = 1.0 + sum_n;

    for (std::size_t i = 0; i < n; ++i) {
        const double visit = i < visits.size() ? visits[i] : 0.0;
        scores[i] -= (visit / denom);
    }
}

void compute_gumbel_scores_into(
    const NodeArena& arena,
    const int node_index,
    std::vector<double>& scores) {
    compute_gumbel_scores_with_policy_into(
        arena, node_index, arena.node(node_index).child_priors(), scores);
}

void compute_least_visited_scores_into(
    const NodeArena& arena,
    const int node_index,
    std::vector<double>& scores) {
    const Node& node = arena.node(node_index);
    const std::size_t n = child_count(node);
    scores.assign(n, 0.0);
    const std::vector<double>& visits = node.child_visits();
    for (std::size_t i = 0; i < n; ++i) {
        const double v = i < visits.size() ? visits[i] : 0.0;
        scores[i] = -v;
    }
}

void compute_prior_scores_into(
    const NodeArena& arena,
    const int node_index,
    std::vector<double>& scores) {
    const Node& node = arena.node(node_index);
    copy_or_pad(node.child_priors(), child_count(node), scores);
}

void compute_q_value_scores_into(
    const NodeArena& arena,
    const int node_index,
    std::vector<double>& scores) {
    const Node& node = arena.node(node_index);
    copy_or_pad(node.child_values(), child_count(node), scores);
}

}  // namespace

double score_initial(const ScoringMethodType type, const double prior, int /*action*/) {
    switch (type) {
        case ScoringMethodType::kUcb:
        case ScoringMethodType::kGumbel:
        case ScoringMethodType::kPrior:
        case ScoringMethodType::kQValue:
            return prior;
        case ScoringMethodType::kLeastVisited:
            return 0.0;
    }
    return prior;
}

std::vector<double> compute_ucb_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config) {
    std::vector<double> scores;
    compute_ucb_scores_into(arena, node_index, min_max_stats, config, scores);
    return scores;
}

std::vector<double> compute_gumbel_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    std::vector<double> scores;
    compute_gumbel_scores_into(arena, node_index, scores);
    return scores;
}

std::vector<double> compute_gumbel_scores_with_policy(
    const NodeArena& arena,
    const int node_index,
    const std::vector<double>& improved_policy,
    const ScoringConfig& /*config*/) {
    std::vector<double> scores;
    compute_gumbel_scores_with_policy_into(arena, node_index, improved_policy, scores);
    return scores;
}

std::vector<double> compute_least_visited_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    std::vector<double> scores;
    compute_least_visited_scores_into(arena, node_index, scores);
    return scores;
}

std::vector<double> compute_prior_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    std::vector<double> scores;
    compute_prior_scores_into(arena, node_index, scores);
    return scores;
}

std::vector<double> compute_q_value_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    std::vector<double> scores;
    compute_q_value_scores_into(arena, node_index, scores);
    return scores;
}

ScoreFunction resolve_scoring_function(const ScoringMethodType type) {
    switch (type) {
        case ScoringMethodType::kUcb:
            return &compute_ucb_scores;
        case ScoringMethodType::kGumbel:
            return &compute_gumbel_scores;
        case ScoringMethodType::kLeastVisited:
            return &compute_least_visited_scores;
        case ScoringMethodType::kPrior:
            return &compute_prior_scores;
        case ScoringMethodType::kQValue:
            return &compute_q_value_scores;
    }
    throw std::invalid_argument("Unsupported scoring method.");
}

std::vector<double> compute_scores(
    const ScoringMethodType type,
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config) {
    std::vector<double> scores;
    compute_scores(type, arena, node_index, min_max_stats, config, scores);
    return scores;
}

void compute_scores(
    const ScoringMethodType type,
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& min_max_stats,
    const ScoringConfig& config,
    std::vector<double>& out_scores) {
    switch (type) {
        case ScoringMethodType::kUcb:
            compute_ucb_scores_into(arena, node_index, min_max_stats, config, out_scores);
            return;
        case ScoringMethodType::kGumbel:
            compute_gumbel_scores_into(arena, node_index, out_scores);
            return;
        case ScoringMethodType::kLeastVisited:
            compute_least_visited_scores_into(arena, node_index, out_scores);
            return;
        case ScoringMethodType::kPrior:
            compute_prior_scores_into(arena, node_index, out_scores);
            return;
        case ScoringMethodType::kQValue:
            compute_q_value_scores_into(arena, node_index, out_scores);
            return;
    }
    throw std::invalid_argument("Unsupported scoring method.");
}

}  // namespace search
