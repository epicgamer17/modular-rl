#include "selection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace search {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

int draw_from_weights(const std::vector<double>& weights, std::mt19937_64& rng) {
    double sum = 0.0;
    for (const double w : weights) {
        if (w > 0.0 && std::isfinite(w)) {
            sum += w;
        }
    }
    if (sum <= 0.0) {
        return -1;
    }

    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng);
    double cdf = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double w = (weights[i] > 0.0 && std::isfinite(weights[i])) ? weights[i] : 0.0;
        cdf += w;
        if (r <= cdf) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(weights.size()) - 1;
}

std::vector<int> top_indices(const std::vector<double>& scores, const double best_value) {
    constexpr double kTieEpsilon = 1e-12;
    std::vector<int> out;
    out.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (std::isfinite(scores[i]) && std::abs(scores[i] - best_value) <= kTieEpsilon) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

int random_choice(const std::vector<int>& candidates, std::mt19937_64& rng) {
    if (candidates.empty()) {
        return -1;
    }
    if (candidates.size() == 1) {
        return candidates.front();
    }
    std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
    return candidates[static_cast<std::size_t>(dist(rng))];
}

}  // namespace

std::vector<double> mask_actions(
    const std::vector<double>& values,
    const std::vector<int>& legal_moves,
    const double mask_value) {
    if (values.empty()) {
        return {};
    }

    std::vector<double> masked(values.size(), mask_value);
    for (const int move : legal_moves) {
        if (move < 0 || static_cast<std::size_t>(move) >= values.size()) {
            continue;
        }
        masked[static_cast<std::size_t>(move)] = values[static_cast<std::size_t>(move)];
    }
    return masked;
}

int select_top_score(
    const std::vector<double>& scores,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    if (scores.empty()) {
        return -1;
    }

    const auto best_it = std::max_element(scores.begin(), scores.end());
    if (best_it == scores.end() || !std::isfinite(*best_it)) {
        return static_cast<int>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
    }

    const double best = *best_it;
    const std::vector<int> ties = top_indices(scores, best);
    if (!config.random_tiebreak || ties.size() <= 1) {
        return ties.empty() ? static_cast<int>(std::distance(scores.begin(), best_it)) : ties.front();
    }
    return random_choice(ties, rng);
}

int select_top_score_with_tiebreak(
    const std::vector<double>& scores,
    const std::vector<double>& tiebreak_scores,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    if (scores.empty()) {
        return -1;
    }
    const int top = select_top_score(scores, config, rng);
    if (top < 0 || tiebreak_scores.size() != scores.size()) {
        return top;
    }

    const double top_value = scores[static_cast<std::size_t>(top)];
    const std::vector<int> ties = top_indices(scores, top_value);
    if (ties.size() <= 1) {
        return top;
    }

    double best_tiebreak = kNegInf;
    for (const int idx : ties) {
        best_tiebreak = std::max(best_tiebreak, tiebreak_scores[static_cast<std::size_t>(idx)]);
    }

    std::vector<int> second_ties;
    second_ties.reserve(ties.size());
    constexpr double kTieEpsilon = 1e-12;
    for (const int idx : ties) {
        if (std::abs(tiebreak_scores[static_cast<std::size_t>(idx)] - best_tiebreak) <= kTieEpsilon) {
            second_ties.push_back(idx);
        }
    }
    if (!config.random_tiebreak || second_ties.size() <= 1) {
        return second_ties.front();
    }
    return random_choice(second_ties, rng);
}

int sample_from_softmax(
    const std::vector<double>& logits,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    if (logits.empty()) {
        return -1;
    }
    if (config.temperature <= 0.0) {
        return select_top_score(logits, config, rng);
    }

    const double inv_temp = 1.0 / config.temperature;
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size(), 0.0);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        weights[i] = std::exp((logits[i] - max_logit) * inv_temp);
    }
    const int sampled = draw_from_weights(weights, rng);
    return sampled >= 0 ? sampled : select_top_score(logits, config, rng);
}

int sample_from_probabilities(
    const std::vector<double>& probs,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    if (probs.empty()) {
        return -1;
    }
    if (config.temperature <= 0.0) {
        return select_top_score(probs, config, rng);
    }

    const double exponent = 1.0 / config.temperature;
    std::vector<double> weights(probs.size(), 0.0);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        const double p = std::max(0.0, probs[i]);
        weights[i] = std::pow(std::max(p, 1e-12), exponent);
    }
    const int sampled = draw_from_weights(weights, rng);
    return sampled >= 0 ? sampled : select_top_score(probs, config, rng);
}

int select_max_visit_count(
    const Node& node,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    return select_top_score(node.child_visits(), config, rng);
}

int select_chance_outcome(
    const ChanceNode& node,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    if (!node.child_priors().empty()) {
        return sample_from_probabilities(node.child_priors(), config, rng);
    }
    std::vector<double> probs;
    probs.reserve(node.code_probs().size());
    int max_code = -1;
    for (const auto& [code, _] : node.code_probs()) {
        max_code = std::max(max_code, code);
    }
    if (max_code < 0) {
        return -1;
    }
    probs.assign(static_cast<std::size_t>(max_code) + 1, 0.0);
    for (const auto& [code, prob] : node.code_probs()) {
        probs[static_cast<std::size_t>(code)] = prob;
    }
    return sample_from_probabilities(probs, config, rng);
}

SelectionFunction resolve_selection_function(const SelectionMethodType type) {
    switch (type) {
        case SelectionMethodType::kTopScore:
            return &select_top_score;
        case SelectionMethodType::kSoftmaxSample:
            return &sample_from_softmax;
        case SelectionMethodType::kProbabilitySample:
        case SelectionMethodType::kMaxVisit:
            return &sample_from_probabilities;
    }
    throw std::invalid_argument("Unsupported selection method.");
}

int select_action(
    const SelectionMethodType type,
    const std::vector<double>& scores,
    const SelectionConfig& config,
    std::mt19937_64& rng) {
    const SelectionFunction fn = resolve_selection_function(type);
    return fn(scores, config, rng);
}

}  // namespace search
