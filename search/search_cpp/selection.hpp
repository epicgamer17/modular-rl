#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "nodes.hpp"

namespace search {

enum class SelectionMethodType {
    kTopScore = 0,
    kSoftmaxSample = 1,
    kProbabilitySample = 2,
    kMaxVisit = 3,
};

struct SelectionConfig {
    double temperature = 1.0;
    bool random_tiebreak = true;
    uint64_t seed = 0;
    double mask_value = -1e30;
};

using SelectionFunction = int (*)(
    const std::vector<double>& scores,
    const SelectionConfig& config,
    std::mt19937_64& rng);

std::vector<double> mask_actions(
    const std::vector<double>& values,
    const std::vector<int>& legal_moves,
    double mask_value = -1e30);

int select_top_score(
    const std::vector<double>& scores,
    const SelectionConfig& config,
    std::mt19937_64& rng);

int select_top_score_with_tiebreak(
    const std::vector<double>& scores,
    const std::vector<double>& tiebreak_scores,
    const SelectionConfig& config,
    std::mt19937_64& rng);

int sample_from_softmax(
    const std::vector<double>& logits,
    const SelectionConfig& config,
    std::mt19937_64& rng);

int sample_from_probabilities(
    const std::vector<double>& probs,
    const SelectionConfig& config,
    std::mt19937_64& rng);

int select_max_visit_count(
    const Node& node,
    const SelectionConfig& config,
    std::mt19937_64& rng);

int select_chance_outcome(
    const ChanceNode& node,
    const SelectionConfig& config,
    std::mt19937_64& rng);

SelectionFunction resolve_selection_function(SelectionMethodType type);

int select_action(
    SelectionMethodType type,
    const std::vector<double>& scores,
    const SelectionConfig& config,
    std::mt19937_64& rng);

}  // namespace search
