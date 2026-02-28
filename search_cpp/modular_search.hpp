#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "backprop.hpp"
#include "min_max_stats.hpp"
#include "nodes.hpp"
#include "scoring.hpp"
#include "selection.hpp"

namespace search {

enum class InferenceKind {
    kHiddenState = 0,
    kAfterstate = 1,
};

struct SearchConfig {
    int num_actions = 0;
    int num_players = 2;
    int default_batch_size = 1;
    bool stochastic = true;

    std::vector<double> known_bounds;
    bool soft_update_minmax = false;
    double min_max_epsilon = 0.01;

    double discount_factor = 1.0;
    bool alternating_minimax = false;
    int perspective_player = -1;
};

struct LeafBatchRequest {
    std::vector<int32_t> hidden_request_ids;
    std::vector<int64_t> hidden_parent_state_handles;
    std::vector<int32_t> hidden_actions;
    std::vector<uint8_t> hidden_action_is_one_hot;
    std::vector<int32_t> hidden_num_codes;

    std::vector<int32_t> afterstate_request_ids;
    std::vector<int64_t> afterstate_parent_state_handles;
    std::vector<int32_t> afterstate_actions;

    std::size_t total_size() const;
    bool empty() const;
};

struct HiddenInferenceUpdateBatch {
    std::vector<int32_t> request_ids;
    std::vector<int64_t> next_state_handles;
    std::vector<double> rewards;
    std::vector<double> values;
    std::vector<int32_t> to_plays;
    std::vector<double> priors;  // row-major: [batch, num_actions]
    int num_actions = 0;
};

struct AfterstateInferenceUpdateBatch {
    std::vector<int32_t> request_ids;
    std::vector<int64_t> next_state_handles;
    std::vector<double> values;
    std::vector<double> code_probs;  // row-major: [batch, num_codes]
    int num_codes = 0;
};

class SearchAlgorithm {
public:
    SearchAlgorithm(
        const SearchConfig& search_config,
        const ScoringConfig& scoring_config,
        const SelectionConfig& root_selection_config,
        const SelectionConfig& decision_selection_config,
        const SelectionConfig& chance_selection_config,
        const BackpropConfig& backprop_config,
        ScoringMethodType root_scoring_type = ScoringMethodType::kUcb,
        ScoringMethodType decision_scoring_type = ScoringMethodType::kUcb,
        SelectionMethodType root_selection_type = SelectionMethodType::kTopScore,
        SelectionMethodType decision_selection_type = SelectionMethodType::kTopScore,
        SelectionMethodType chance_selection_type = SelectionMethodType::kProbabilitySample,
        BackpropMethodType backprop_method = BackpropMethodType::kAverageDiscountedReturn);

    void clear();
    bool has_root() const;
    int root_index() const;
    std::size_t node_count() const;
    std::size_t pending_count() const;

    int initialize_root(
        const std::vector<double>& policy_priors,
        int to_play,
        int64_t state_handle,
        double root_value = 0.0,
        const std::vector<int>& allowed_actions = {});

    LeafBatchRequest step_search_until_leaves(int batch_size = -1);

    int update_leaves_and_backprop(
        const HiddenInferenceUpdateBatch& hidden_updates,
        const AfterstateInferenceUpdateBatch& afterstate_updates);

    double root_value() const;
    std::vector<double> root_child_priors() const;
    std::vector<double> root_child_values() const;
    std::vector<double> root_child_visits() const;

    int select_root_action(SelectionMethodType method = SelectionMethodType::kMaxVisit);

    const MinMaxStats& min_max_stats() const;
    MinMaxStats& mutable_min_max_stats();
    const NodeArena& arena() const;
    NodeArena& mutable_arena();

private:
    struct PendingSimulation {
        bool active = false;
        int leaf_index = -1;
        int parent_index = -1;
        int action = -1;
        InferenceKind inference_kind = InferenceKind::kHiddenState;
        bool action_is_one_hot = false;
        int num_codes = 0;
        int parent_to_play = -1;
        std::vector<int> search_path;
        std::vector<int> action_path;
    };

    PendingSimulation build_single_pending_simulation();
    int select_action_for_node(int node_index, bool is_root_node);
    int ensure_child_for_edge(int parent_index, int action);
    void apply_hidden_update(const HiddenInferenceUpdateBatch& updates, int i);
    void apply_afterstate_update(const AfterstateInferenceUpdateBatch& updates, int i);
    PendingSimulation& pending_or_throw(int request_id);

    SearchConfig search_config_;
    ScoringConfig scoring_config_;
    SelectionConfig root_selection_config_;
    SelectionConfig decision_selection_config_;
    SelectionConfig chance_selection_config_;
    BackpropConfig backprop_config_;

    ScoringMethodType root_scoring_type_;
    ScoringMethodType decision_scoring_type_;
    SelectionMethodType root_selection_type_;
    SelectionMethodType decision_selection_type_;
    SelectionMethodType chance_selection_type_;
    BackpropMethodType backprop_method_;

    NodeArena arena_;
    MinMaxStats min_max_stats_;
    int root_index_;

    std::vector<PendingSimulation> pending_;
    std::size_t active_pending_count_;

    std::mt19937_64 rng_;
};

}  // namespace search
