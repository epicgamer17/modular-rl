#include "scoring.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace rainbow::search {

namespace {

std::size_t child_count(const Node& node) {
    return node.child_priors().size();
}

std::vector<double> fallback_copy(const std::vector<double>& source, const std::size_t n) {
    std::vector<double> out(n, 0.0);
    const std::size_t k = std::min(n, source.size());
    for (std::size_t i = 0; i < k; ++i) {
        out[i] = source[i];
    }
    return out;
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
    const Node& node = arena.node(node_index);
    const std::size_t n = child_count(node);
    if (n == 0) {
        return {};
    }

    const std::vector<double>& priors = node.child_priors();
    const std::vector<double>& visits = node.child_visits();
    const std::vector<double>& values = node.child_values();

    const double pb_c_base = std::max(config.pb_c_base, 1e-12);
    double pb_c = std::log((static_cast<double>(node.visits()) + pb_c_base + 1.0) / pb_c_base) + config.pb_c_init;
    pb_c *= std::sqrt(std::max(0.0, static_cast<double>(node.visits())));

    std::vector<double> scores(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double child_visit = i < visits.size() ? visits[i] : 0.0;
        const double prior = i < priors.size() ? priors[i] : 0.0;
        const double q = (child_visit > 0.0 && i < values.size()) ? values[i] : config.unvisited_value_bootstrap;
        const double prior_score = (pb_c / (child_visit + 1.0)) * prior;
        const double value_score = min_max_stats.normalize(q);
        scores[i] = prior_score + value_score;
    }
    return scores;
}

std::vector<double> compute_gumbel_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& config) {
    const Node& node = arena.node(node_index);
    return compute_gumbel_scores_with_policy(arena, node_index, node.child_priors(), config);
}

std::vector<double> compute_gumbel_scores_with_policy(
    const NodeArena& arena,
    const int node_index,
    const std::vector<double>& improved_policy,
    const ScoringConfig& /*config*/) {
    const Node& node = arena.node(node_index);
    const std::size_t n = child_count(node);
    if (n == 0) {
        return {};
    }

    const std::vector<double>& visits = node.child_visits();
    const std::vector<double> pi0 = fallback_copy(improved_policy.empty() ? node.child_priors() : improved_policy, n);
    const double sum_n = std::accumulate(visits.begin(), visits.end(), 0.0);
    const double denom = 1.0 + sum_n;

    std::vector<double> scores(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double visit = i < visits.size() ? visits[i] : 0.0;
        scores[i] = pi0[i] - (visit / denom);
    }
    return scores;
}

std::vector<double> compute_least_visited_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    const Node& node = arena.node(node_index);
    const std::size_t n = child_count(node);
    std::vector<double> scores(n, 0.0);
    const std::vector<double>& visits = node.child_visits();
    for (std::size_t i = 0; i < n; ++i) {
        const double v = i < visits.size() ? visits[i] : 0.0;
        scores[i] = -v;
    }
    return scores;
}

std::vector<double> compute_prior_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    const Node& node = arena.node(node_index);
    return node.child_priors();
}

std::vector<double> compute_q_value_scores(
    const NodeArena& arena,
    const int node_index,
    const MinMaxStats& /*min_max_stats*/,
    const ScoringConfig& /*config*/) {
    const Node& node = arena.node(node_index);
    return node.child_values();
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
    const ScoreFunction fn = resolve_scoring_function(type);
    return fn(arena, node_index, min_max_stats, config);
}

}  // namespace rainbow::search
