#include "modular_search.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace search {

std::size_t LeafBatchRequest::total_size() const {
    return hidden_request_ids.size() + afterstate_request_ids.size();
}

bool LeafBatchRequest::empty() const {
    return total_size() == 0;
}

SearchAlgorithm::SearchAlgorithm(
    const SearchConfig& search_config,
    const ScoringConfig& scoring_config,
    const SelectionConfig& root_selection_config,
    const SelectionConfig& decision_selection_config,
    const SelectionConfig& chance_selection_config,
    const BackpropConfig& backprop_config,
    const ScoringMethodType root_scoring_type,
    const ScoringMethodType decision_scoring_type,
    const SelectionMethodType root_selection_type,
    const SelectionMethodType decision_selection_type,
    const SelectionMethodType chance_selection_type,
    const BackpropMethodType backprop_method)
    : search_config_(search_config),
      scoring_config_(scoring_config),
      root_selection_config_(root_selection_config),
      decision_selection_config_(decision_selection_config),
      chance_selection_config_(chance_selection_config),
      backprop_config_(backprop_config),
      root_scoring_type_(root_scoring_type),
      decision_scoring_type_(decision_scoring_type),
      root_selection_type_(root_selection_type),
      decision_selection_type_(decision_selection_type),
      chance_selection_type_(chance_selection_type),
      backprop_method_(backprop_method),
      min_max_stats_(
          search_config.known_bounds,
          search_config.soft_update_minmax,
          search_config.min_max_epsilon),
      root_index_(-1),
      active_pending_count_(0) {
    if (search_config_.num_actions < 0) {
        throw std::invalid_argument("num_actions must be non-negative.");
    }

    if (backprop_config_.num_players <= 0) {
        backprop_config_.num_players = std::max(1, search_config_.num_players);
    }
    backprop_config_.discount_factor = search_config_.discount_factor;
    backprop_config_.alternating_minimax = search_config_.alternating_minimax;
    backprop_config_.perspective_player = search_config_.perspective_player;

    std::random_device rd;
    const uint64_t seed = root_selection_config_.seed == 0
        ? (static_cast<uint64_t>(rd()) << 32U) ^ rd()
        : root_selection_config_.seed;
    rng_.seed(seed);

    const std::size_t initial_task_capacity = search_config_.default_batch_size > 0
        ? static_cast<std::size_t>(search_config_.default_batch_size)
        : 0U;
    task_buffer_.reserve(initial_task_capacity);
}

void SearchAlgorithm::clear() {
    arena_.clear();
    pending_.clear();
    task_buffer_.clear();
    active_pending_count_ = 0;
    root_index_ = -1;
    min_max_stats_ = MinMaxStats(
        search_config_.known_bounds,
        search_config_.soft_update_minmax,
        search_config_.min_max_epsilon);
}

bool SearchAlgorithm::has_root() const {
    return root_index_ >= 0;
}

int SearchAlgorithm::root_index() const {
    return root_index_;
}

std::size_t SearchAlgorithm::node_count() const {
    return arena_.size();
}

std::size_t SearchAlgorithm::pending_count() const {
    return active_pending_count_;
}

int SearchAlgorithm::initialize_root(
    const std::vector<double>& policy_priors,
    const int to_play,
    const int64_t state_handle,
    const double root_value,
    const std::vector<int>& allowed_actions) {
    if (policy_priors.empty()) {
        throw std::invalid_argument("Root priors must be non-empty.");
    }
    if (search_config_.num_actions > 0 && static_cast<int>(policy_priors.size()) != search_config_.num_actions) {
        throw std::invalid_argument("Root priors size does not match search_config.num_actions.");
    }
    if (search_config_.num_actions == 0) {
        search_config_.num_actions = static_cast<int>(policy_priors.size());
    }

    clear();
    root_index_ = arena_.create_decision(0.0, -1);
    DecisionNode& root = arena_.decision(root_index_);
    root.set_stochastic(search_config_.stochastic);
    root.set_to_play(to_play);
    root.set_state_handle(state_handle);
    root.set_visits(1);
    root.set_value_sum(root_value);

    root.expand(
        to_play,
        policy_priors,
        policy_priors,
        allowed_actions,
        0.0,
        root_value);

    min_max_stats_.update(root_value);
    return root_index_;
}

SearchAlgorithm::PendingSimulation SearchAlgorithm::build_single_pending_simulation() {
    PendingSimulation pending;
    if (!has_root()) {
        return pending;
    }

    int node_index = root_index_;
    pending.search_path.push_back(node_index);
    constexpr int kMaxDepth = 4096;

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        const Node& node = arena_.node(node_index);
        if (!node.expanded()) {
            break;
        }

        const bool is_root_node = node_index == root_index_;
        const int action = select_action_for_node(node_index, is_root_node);
        if (action < 0) {
            return PendingSimulation{};
        }

        const int child_index = ensure_child_for_edge(node_index, action);
        pending.action_path.push_back(action);
        pending.search_path.push_back(child_index);
        node_index = child_index;
    }

    if (pending.search_path.size() < 2 || pending.action_path.empty()) {
        return PendingSimulation{};
    }

    pending.active = true;
    pending.leaf_index = pending.search_path.back();
    pending.parent_index = pending.search_path[pending.search_path.size() - 2];
    pending.action = pending.action_path.back();
    pending.parent_to_play = arena_.node(pending.parent_index).to_play();

    const Node& leaf = arena_.node(pending.leaf_index);
    const Node& parent = arena_.node(pending.parent_index);
    if (leaf.is_chance()) {
        pending.inference_kind = InferenceKind::kAfterstate;
        pending.action_is_one_hot = false;
        pending.num_codes = 0;
    } else {
        pending.inference_kind = InferenceKind::kHiddenState;
        pending.action_is_one_hot = parent.is_chance();
        pending.num_codes = pending.action_is_one_hot
            ? static_cast<int>(parent.child_priors().size())
            : 0;
    }
    return pending;
}

int SearchAlgorithm::select_action_for_node(const int node_index, const bool is_root_node) {
    const Node& node = arena_.node(node_index);
    if (node.is_decision()) {
        const ScoringMethodType scoring_type = is_root_node ? root_scoring_type_ : decision_scoring_type_;
        const SelectionMethodType selection_type = is_root_node ? root_selection_type_ : decision_selection_type_;
        const SelectionConfig& selection_cfg = is_root_node ? root_selection_config_ : decision_selection_config_;
        const std::vector<double> scores = compute_scores(scoring_type, arena_, node_index, min_max_stats_, scoring_config_);
        int action = select_action(selection_type, scores, selection_cfg, rng_);
        if (action < 0 && !scores.empty()) {
            action = static_cast<int>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
        }
        return action;
    }

    const ChanceNode& chance = arena_.chance(node_index);
    switch (chance_selection_type_) {
        case SelectionMethodType::kTopScore:
            return select_top_score(chance.child_priors(), chance_selection_config_, rng_);
        case SelectionMethodType::kSoftmaxSample:
            return sample_from_softmax(chance.child_priors(), chance_selection_config_, rng_);
        case SelectionMethodType::kProbabilitySample:
        case SelectionMethodType::kMaxVisit:
            return select_chance_outcome(chance, chance_selection_config_, rng_);
    }
    return -1;
}

int SearchAlgorithm::ensure_child_for_edge(const int parent_index, const int action) {
    Node& parent = arena_.node(parent_index);
    if (parent.has_child(action)) {
        return parent.child_index(action);
    }

    const std::vector<double>& child_priors = parent.child_priors();
    if (action < 0 || static_cast<std::size_t>(action) >= child_priors.size()) {
        throw std::out_of_range("Action index is outside child priors range.");
    }

    const double prior = child_priors[static_cast<std::size_t>(action)];
    int child_index = -1;
    if (parent.is_decision()) {
        const DecisionNode& decision_parent = arena_.decision(parent_index);
        if (decision_parent.stochastic()) {
            child_index = arena_.create_chance(prior, parent_index);
        } else {
            child_index = arena_.create_decision(prior, parent_index);
            arena_.decision(child_index).set_stochastic(search_config_.stochastic);
        }
    } else {
        child_index = arena_.create_decision(prior, parent_index);
        arena_.decision(child_index).set_stochastic(search_config_.stochastic);
    }

    parent.set_child(action, child_index);
    return child_index;
}

LeafBatchRequest SearchAlgorithm::step_search_until_leaves(int batch_size) {
    if (!has_root()) {
        throw std::logic_error("SearchAlgorithm has no root. Call initialize_root first.");
    }
    if (active_pending_count_ > 0) {
        throw std::logic_error("Cannot step search while pending leaves are unresolved.");
    }

    if (batch_size <= 0) {
        batch_size = std::max(1, search_config_.default_batch_size);
    }

    LeafBatchRequest request;
    request.hidden_request_ids.reserve(static_cast<std::size_t>(batch_size));
    request.afterstate_request_ids.reserve(static_cast<std::size_t>(batch_size));

    for (int i = 0; i < batch_size; ++i) {
        PendingSimulation pending = build_single_pending_simulation();
        if (!pending.active) {
            break;
        }

        const int request_id = static_cast<int>(pending_.size());
        pending_.push_back(std::move(pending));
        ++active_pending_count_;
        PendingSimulation& stored = pending_.back();

        const int64_t parent_state = arena_.node(stored.parent_index).state_handle();
        if (stored.inference_kind == InferenceKind::kHiddenState) {
            request.hidden_request_ids.push_back(request_id);
            request.hidden_parent_state_handles.push_back(parent_state);
            request.hidden_actions.push_back(stored.action);
            request.hidden_action_is_one_hot.push_back(stored.action_is_one_hot ? 1U : 0U);
            request.hidden_num_codes.push_back(stored.num_codes);
        } else {
            request.afterstate_request_ids.push_back(request_id);
            request.afterstate_parent_state_handles.push_back(parent_state);
            request.afterstate_actions.push_back(stored.action);
        }
    }

    return request;
}

SearchAlgorithm::PendingSimulation& SearchAlgorithm::pending_or_throw(const int request_id) {
    if (request_id < 0 || static_cast<std::size_t>(request_id) >= pending_.size()) {
        throw std::out_of_range("Invalid request_id.");
    }
    PendingSimulation& pending = pending_[static_cast<std::size_t>(request_id)];
    if (!pending.active) {
        throw std::logic_error("request_id is not active.");
    }
    return pending;
}

void SearchAlgorithm::apply_hidden_update(const HiddenInferenceUpdateBatch& updates, const int i) {
    const std::size_t row_start = static_cast<std::size_t>(i) * static_cast<std::size_t>(updates.num_actions);
    const double* priors_row = updates.num_actions > 0
        ? updates.priors.data() + row_start
        : nullptr;
    apply_hidden_update_raw(
        updates.request_ids[static_cast<std::size_t>(i)],
        updates.next_state_handles[static_cast<std::size_t>(i)],
        updates.rewards[static_cast<std::size_t>(i)],
        updates.values[static_cast<std::size_t>(i)],
        updates.to_plays[static_cast<std::size_t>(i)],
        priors_row,
        updates.num_actions);
}

void SearchAlgorithm::apply_afterstate_update(const AfterstateInferenceUpdateBatch& updates, const int i) {
    const std::size_t row_start = static_cast<std::size_t>(i) * static_cast<std::size_t>(updates.num_codes);
    const double* code_probs_row = updates.num_codes > 0
        ? updates.code_probs.data() + row_start
        : nullptr;
    apply_afterstate_update_raw(
        updates.request_ids[static_cast<std::size_t>(i)],
        updates.next_state_handles[static_cast<std::size_t>(i)],
        updates.values[static_cast<std::size_t>(i)],
        code_probs_row,
        updates.num_codes);
}

void SearchAlgorithm::apply_hidden_update_raw(
    const int32_t request_id,
    const int64_t next_state_handle,
    const double reward,
    const double value,
    const int32_t to_play,
    const double* priors_row,
    const int num_actions) {
    PendingSimulation& pending = pending_or_throw(request_id);
    if (pending.inference_kind != InferenceKind::kHiddenState) {
        throw std::logic_error("Received hidden-state update for a non-hidden request.");
    }

    DecisionNode& leaf = arena_.decision(pending.leaf_index);
    leaf.expand_dense(to_play, priors_row, num_actions, reward, value);
    leaf.set_stochastic(search_config_.stochastic);
    leaf.set_state_handle(next_state_handle);

    backpropagate_with_method(
        backprop_method_,
        arena_,
        pending.search_path,
        pending.action_path,
        value,
        to_play,
        min_max_stats_,
        backprop_config_);

    pending.active = false;
    --active_pending_count_;
}

void SearchAlgorithm::apply_afterstate_update_raw(
    const int32_t request_id,
    const int64_t next_state_handle,
    const double value,
    const double* code_probs_row,
    const int num_codes) {
    PendingSimulation& pending = pending_or_throw(request_id);
    if (pending.inference_kind != InferenceKind::kAfterstate) {
        throw std::logic_error("Received afterstate update for a non-afterstate request.");
    }

    ChanceNode& leaf = arena_.chance(pending.leaf_index);
    leaf.expand_dense(pending.parent_to_play, value, code_probs_row, num_codes);
    leaf.set_state_handle(next_state_handle);

    backpropagate_with_method(
        backprop_method_,
        arena_,
        pending.search_path,
        pending.action_path,
        value,
        pending.parent_to_play,
        min_max_stats_,
        backprop_config_);

    pending.active = false;
    --active_pending_count_;
}

int SearchAlgorithm::update_leaves_and_backprop(
    const HiddenInferenceUpdateBatch& hidden_updates,
    const AfterstateInferenceUpdateBatch& afterstate_updates) {
    const std::size_t hidden_n = hidden_updates.request_ids.size();
    const std::size_t afterstate_n = afterstate_updates.request_ids.size();
    if (hidden_n != hidden_updates.next_state_handles.size() ||
        hidden_n != hidden_updates.rewards.size() ||
        hidden_n != hidden_updates.values.size() ||
        hidden_n != hidden_updates.to_plays.size()) {
        throw std::invalid_argument(
            "Hidden update batch fields must have equal lengths.");
    }
    if (hidden_updates.num_actions < 0) {
        throw std::invalid_argument("hidden_updates.num_actions must be non-negative.");
    }
    if (static_cast<std::size_t>(hidden_updates.num_actions) * hidden_n !=
        hidden_updates.priors.size()) {
        throw std::invalid_argument(
            "Hidden priors matrix shape does not match [batch, num_actions].");
    }
    if (afterstate_n != afterstate_updates.next_state_handles.size() ||
        afterstate_n != afterstate_updates.values.size()) {
        throw std::invalid_argument(
            "Afterstate update batch fields must have equal lengths.");
    }
    if (afterstate_updates.num_codes < 0) {
        throw std::invalid_argument(
            "afterstate_updates.num_codes must be non-negative.");
    }
    if (static_cast<std::size_t>(afterstate_updates.num_codes) * afterstate_n !=
        afterstate_updates.code_probs.size()) {
        throw std::invalid_argument(
            "Afterstate code_probs matrix shape does not match [batch, num_codes].");
    }

    const int32_t* hidden_request_ids = hidden_n > 0 ? hidden_updates.request_ids.data() : nullptr;
    const int64_t* hidden_next_state_handles = hidden_n > 0 ? hidden_updates.next_state_handles.data() : nullptr;
    const double* hidden_rewards = hidden_n > 0 ? hidden_updates.rewards.data() : nullptr;
    const double* hidden_values = hidden_n > 0 ? hidden_updates.values.data() : nullptr;
    const int32_t* hidden_to_plays = hidden_n > 0 ? hidden_updates.to_plays.data() : nullptr;
    const double* hidden_priors = hidden_n > 0 ? hidden_updates.priors.data() : nullptr;

    const int32_t* afterstate_request_ids = afterstate_n > 0 ? afterstate_updates.request_ids.data() : nullptr;
    const int64_t* afterstate_next_state_handles = afterstate_n > 0 ? afterstate_updates.next_state_handles.data() : nullptr;
    const double* afterstate_values = afterstate_n > 0 ? afterstate_updates.values.data() : nullptr;
    const double* afterstate_code_probs = afterstate_n > 0 ? afterstate_updates.code_probs.data() : nullptr;

    return update_leaves_and_backprop_raw(
        hidden_request_ids,
        hidden_next_state_handles,
        hidden_rewards,
        hidden_values,
        hidden_to_plays,
        hidden_priors,
        hidden_n,
        hidden_updates.num_actions,
        afterstate_request_ids,
        afterstate_next_state_handles,
        afterstate_values,
        afterstate_code_probs,
        afterstate_n,
        afterstate_updates.num_codes);
}

int SearchAlgorithm::update_leaves_and_backprop_raw(
    const int32_t* hidden_request_ids,
    const int64_t* hidden_next_state_handles,
    const double* hidden_rewards,
    const double* hidden_values,
    const int32_t* hidden_to_plays,
    const double* hidden_priors,
    const std::size_t hidden_count,
    const int hidden_num_actions,
    const int32_t* afterstate_request_ids,
    const int64_t* afterstate_next_state_handles,
    const double* afterstate_values,
    const double* afterstate_code_probs,
    const std::size_t afterstate_count,
    const int afterstate_num_codes) {
    if (hidden_num_actions < 0) {
        throw std::invalid_argument("hidden_num_actions must be non-negative.");
    }
    if (afterstate_num_codes < 0) {
        throw std::invalid_argument("afterstate_num_codes must be non-negative.");
    }
    if (hidden_count > 0 && hidden_num_actions <= 0) {
        throw std::invalid_argument("hidden_num_actions must be > 0 when hidden_count > 0.");
    }
    if (afterstate_count > 0 && afterstate_num_codes <= 0) {
        throw std::invalid_argument("afterstate_num_codes must be > 0 when afterstate_count > 0.");
    }
    if (hidden_count > 0 &&
        (hidden_request_ids == nullptr || hidden_next_state_handles == nullptr ||
         hidden_rewards == nullptr || hidden_values == nullptr || hidden_to_plays == nullptr)) {
        throw std::invalid_argument("Hidden update pointers must not be null when hidden_count > 0.");
    }
    if (afterstate_count > 0 &&
        (afterstate_request_ids == nullptr || afterstate_next_state_handles == nullptr || afterstate_values == nullptr)) {
        throw std::invalid_argument("Afterstate update pointers must not be null when afterstate_count > 0.");
    }
    if (hidden_count > 0 && hidden_num_actions > 0 && hidden_priors == nullptr) {
        throw std::invalid_argument("hidden_priors pointer must not be null when hidden updates are present.");
    }
    if (afterstate_count > 0 && afterstate_num_codes > 0 && afterstate_code_probs == nullptr) {
        throw std::invalid_argument("afterstate_code_probs pointer must not be null when afterstate updates are present.");
    }

    const std::size_t total_tasks = hidden_count + afterstate_count;
    if (task_buffer_.capacity() < total_tasks) {
        task_buffer_.reserve(total_tasks);
    }
    task_buffer_.clear();

    for (std::size_t i = 0; i < hidden_count; ++i) {
        task_buffer_.push_back(UpdateTask{
            hidden_request_ids[i],
            static_cast<int>(i),
            true});
    }
    for (std::size_t i = 0; i < afterstate_count; ++i) {
        task_buffer_.push_back(UpdateTask{
            afterstate_request_ids[i],
            static_cast<int>(i),
            false});
    }

    std::sort(task_buffer_.begin(), task_buffer_.end());

    int processed = 0;
    for (const UpdateTask& task : task_buffer_) {
        if (task.is_hidden) {
            const std::size_t i = static_cast<std::size_t>(task.array_index);
            const double* priors_row = hidden_num_actions > 0
                ? hidden_priors + (i * static_cast<std::size_t>(hidden_num_actions))
                : nullptr;
            apply_hidden_update_raw(
                task.request_id,
                hidden_next_state_handles[i],
                hidden_rewards[i],
                hidden_values[i],
                hidden_to_plays[i],
                priors_row,
                hidden_num_actions);
        } else {
            const std::size_t i = static_cast<std::size_t>(task.array_index);
            const double* code_probs_row = afterstate_num_codes > 0
                ? afterstate_code_probs + (i * static_cast<std::size_t>(afterstate_num_codes))
                : nullptr;
            apply_afterstate_update_raw(
                task.request_id,
                afterstate_next_state_handles[i],
                afterstate_values[i],
                code_probs_row,
                afterstate_num_codes);
        }
        ++processed;
    }

    return processed;
}

double SearchAlgorithm::root_value() const {
    if (!has_root()) {
        throw std::logic_error("No root node initialized.");
    }
    return arena_.node(root_index_).value();
}

std::vector<double> SearchAlgorithm::root_child_priors() const {
    if (!has_root()) {
        return {};
    }
    return arena_.node(root_index_).child_priors();
}

std::vector<double> SearchAlgorithm::root_child_values() const {
    if (!has_root()) {
        return {};
    }
    return arena_.node(root_index_).child_values();
}

std::vector<double> SearchAlgorithm::root_child_visits() const {
    if (!has_root()) {
        return {};
    }
    return arena_.node(root_index_).child_visits();
}

int SearchAlgorithm::select_root_action(const SelectionMethodType method) {
    if (!has_root()) {
        return -1;
    }
    const Node& root = arena_.node(root_index_);
    if (method == SelectionMethodType::kMaxVisit) {
        return select_max_visit_count(root, root_selection_config_, rng_);
    }
    const std::vector<double> scores = compute_scores(
        root_scoring_type_,
        arena_,
        root_index_,
        min_max_stats_,
        scoring_config_);
    return select_action(method, scores, root_selection_config_, rng_);
}

const MinMaxStats& SearchAlgorithm::min_max_stats() const {
    return min_max_stats_;
}

MinMaxStats& SearchAlgorithm::mutable_min_max_stats() {
    return min_max_stats_;
}

const NodeArena& SearchAlgorithm::arena() const {
    return arena_;
}

NodeArena& SearchAlgorithm::mutable_arena() {
    return arena_;
}

}  // namespace search
