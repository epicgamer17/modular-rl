#include "nodes.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace search {

Node::Node(const NodeType node_type, const double prior, const int parent_index)
    : node_type_(node_type),
      index_(-1),
      parent_index_(parent_index),
      visits_(0),
      to_play_(-1),
      state_handle_(-1),
      prior_(prior),
      value_sum_(0.0) {}

NodeType Node::node_type() const {
    return node_type_;
}

bool Node::is_decision() const {
    return node_type_ == NodeType::kDecision;
}

bool Node::is_chance() const {
    return node_type_ == NodeType::kChance;
}

int Node::index() const {
    return index_;
}

void Node::set_index(const int index) {
    index_ = index;
}

int Node::parent_index() const {
    return parent_index_;
}

void Node::set_parent_index(const int parent_index) {
    parent_index_ = parent_index;
}

int Node::visits() const {
    return visits_;
}

void Node::set_visits(const int visits) {
    visits_ = visits;
}

void Node::add_visit(const int delta) {
    visits_ += delta;
}

double Node::prior() const {
    return prior_;
}

void Node::set_prior(const double prior) {
    prior_ = prior;
}

double Node::value_sum() const {
    return value_sum_;
}

void Node::set_value_sum(const double value_sum) {
    value_sum_ = value_sum;
}

void Node::add_value(const double delta) {
    value_sum_ += delta;
}

int Node::to_play() const {
    return to_play_;
}

void Node::set_to_play(const int to_play) {
    to_play_ = to_play;
}

int64_t Node::state_handle() const {
    return state_handle_;
}

void Node::set_state_handle(const int64_t state_handle) {
    state_handle_ = state_handle;
}

double Node::value(const double bootstrap) const {
    if (visits_ == 0) {
        return bootstrap;
    }
    return value_sum_ / static_cast<double>(visits_);
}

bool Node::expanded() const {
    return !child_priors_.empty();
}

void Node::set_child_stats_size(const std::size_t size) {
    child_priors_.assign(size, 0.0);
    child_values_.assign(size, 0.0);
    child_visits_.assign(size, 0.0);
}

const std::vector<double>& Node::child_priors() const {
    return child_priors_;
}

const std::vector<double>& Node::child_values() const {
    return child_values_;
}

const std::vector<double>& Node::child_visits() const {
    return child_visits_;
}

std::vector<double>& Node::mutable_child_priors() {
    return child_priors_;
}

std::vector<double>& Node::mutable_child_values() {
    return child_values_;
}

std::vector<double>& Node::mutable_child_visits() {
    return child_visits_;
}

bool Node::has_child(const int key) const {
    return children_.find(key) != children_.end();
}

int Node::child_index(const int key) const {
    const auto it = children_.find(key);
    if (it == children_.end()) {
        return -1;
    }
    return it->second;
}

void Node::set_child(const int key, const int child_index) {
    children_[key] = child_index;
}

void Node::remove_child(const int key) {
    children_.erase(key);
}

void Node::clear_children() {
    children_.clear();
}

const std::unordered_map<int, int>& Node::children() const {
    return children_;
}

std::vector<int> Node::child_keys() const {
    std::vector<int> keys;
    keys.reserve(children_.size());
    for (const auto& [key, _] : children_) {
        keys.push_back(key);
    }
    return keys;
}

DecisionNode::DecisionNode(const double prior, const int parent_index)
    : Node(NodeType::kDecision, prior, parent_index),
      reward_(0.0),
      network_value_(0.0),
      stochastic_(false),
      has_v_mix_cache_(false),
      v_mix_cache_(0.0) {}

void DecisionNode::expand(
    const int to_play,
    const std::vector<double>& network_policy,
    const std::vector<double>& priors,
    const std::vector<int>& allowed_actions,
    const double reward,
    const double network_value) {
    set_to_play(to_play);
    set_reward(reward);
    set_network_value(network_value);
    clear_v_mix_cache();

    network_policy_ = network_policy;
    std::vector<double> selected_priors = priors.empty() ? network_policy : priors;
    if (selected_priors.empty()) {
        throw std::invalid_argument("DecisionNode::expand requires non-empty policy/prior vector.");
    }

    if (!allowed_actions.empty()) {
        std::vector<bool> allowed_mask(selected_priors.size(), false);
        for (const int action : allowed_actions) {
            if (action < 0 || static_cast<std::size_t>(action) >= selected_priors.size()) {
                throw std::out_of_range("allowed action index is out of range.");
            }
            allowed_mask[static_cast<std::size_t>(action)] = true;
        }

        for (std::size_t i = 0; i < selected_priors.size(); ++i) {
            if (!allowed_mask[i]) {
                selected_priors[i] = 0.0;
            }
        }

        const double sum = std::accumulate(selected_priors.begin(), selected_priors.end(), 0.0);
        if (sum > 0.0) {
            for (double& p : selected_priors) {
                p /= sum;
            }
        } else {
            const double uniform = 1.0 / static_cast<double>(allowed_actions.size());
            std::fill(selected_priors.begin(), selected_priors.end(), 0.0);
            for (const int action : allowed_actions) {
                selected_priors[static_cast<std::size_t>(action)] = uniform;
            }
        }
    }

    set_child_stats_size(selected_priors.size());
    mutable_child_priors() = std::move(selected_priors);
}

void DecisionNode::expand_dense(
    const int to_play,
    const double* priors,
    const int num_actions,
    const double reward,
    const double network_value) {
    if (priors == nullptr) {
        throw std::invalid_argument("DecisionNode::expand_dense priors pointer is null.");
    }
    if (num_actions <= 0) {
        throw std::invalid_argument("DecisionNode::expand_dense requires num_actions > 0.");
    }

    set_to_play(to_play);
    set_reward(reward);
    set_network_value(network_value);
    clear_v_mix_cache();

    network_policy_.assign(priors, priors + num_actions);
    set_child_stats_size(static_cast<std::size_t>(num_actions));
    mutable_child_priors().assign(priors, priors + num_actions);
}

double DecisionNode::reward() const {
    return reward_;
}

void DecisionNode::set_reward(const double reward) {
    reward_ = reward;
}

double DecisionNode::network_value() const {
    return network_value_;
}

void DecisionNode::set_network_value(const double network_value) {
    network_value_ = network_value;
}

bool DecisionNode::stochastic() const {
    return stochastic_;
}

void DecisionNode::set_stochastic(const bool stochastic) {
    stochastic_ = stochastic;
}

bool DecisionNode::has_v_mix_cache() const {
    return has_v_mix_cache_;
}

double DecisionNode::v_mix_cache() const {
    return v_mix_cache_;
}

void DecisionNode::set_v_mix_cache(const double v_mix_cache) {
    v_mix_cache_ = v_mix_cache;
    has_v_mix_cache_ = true;
}

void DecisionNode::clear_v_mix_cache() {
    has_v_mix_cache_ = false;
    v_mix_cache_ = 0.0;
}

ChanceNode::ChanceNode(const double prior, const int parent_index)
    : Node(NodeType::kChance, prior, parent_index), network_value_(0.0) {}

void ChanceNode::expand(
    const int to_play,
    const double network_value,
    const std::vector<double>& code_probs) {
    set_to_play(to_play);
    set_network_value(network_value);
    clear_code_probabilities();

    set_child_stats_size(code_probs.size());
    mutable_child_priors() = code_probs;
    for (std::size_t i = 0; i < code_probs.size(); ++i) {
        code_probs_[static_cast<int>(i)] = code_probs[i];
    }
}

void ChanceNode::expand_dense(
    const int to_play,
    const double network_value,
    const double* code_probs,
    const int num_codes) {
    if (code_probs == nullptr) {
        throw std::invalid_argument("ChanceNode::expand_dense code_probs pointer is null.");
    }
    if (num_codes <= 0) {
        throw std::invalid_argument("ChanceNode::expand_dense requires num_codes > 0.");
    }

    set_to_play(to_play);
    set_network_value(network_value);
    clear_code_probabilities();

    set_child_stats_size(static_cast<std::size_t>(num_codes));
    mutable_child_priors().assign(code_probs, code_probs + num_codes);
    for (int i = 0; i < num_codes; ++i) {
        code_probs_[i] = code_probs[i];
    }
}

double ChanceNode::network_value() const {
    return network_value_;
}

void ChanceNode::set_network_value(const double network_value) {
    network_value_ = network_value;
}

const std::unordered_map<int, double>& ChanceNode::code_probs() const {
    return code_probs_;
}

double ChanceNode::code_probability(const int code) const {
    const auto it = code_probs_.find(code);
    if (it == code_probs_.end()) {
        return 0.0;
    }
    return it->second;
}

void ChanceNode::set_code_probability(const int code, const double prob) {
    code_probs_[code] = prob;
}

void ChanceNode::clear_code_probabilities() {
    code_probs_.clear();
}

int NodeArena::create_decision(const double prior, const int parent_index) {
    decision_nodes_.emplace_back(prior, parent_index);
    const int local_index = static_cast<int>(decision_nodes_.size()) - 1;
    const int global_index = append_entry(NodeType::kDecision, local_index);
    decision_nodes_.back().set_index(global_index);
    return global_index;
}

int NodeArena::create_chance(const double prior, const int parent_index) {
    chance_nodes_.emplace_back(prior, parent_index);
    const int local_index = static_cast<int>(chance_nodes_.size()) - 1;
    const int global_index = append_entry(NodeType::kChance, local_index);
    chance_nodes_.back().set_index(global_index);
    return global_index;
}

void NodeArena::reserve(const std::size_t capacity) {
    entries_.reserve(capacity);
    decision_nodes_.reserve(capacity);
    chance_nodes_.reserve(capacity);
}

bool NodeArena::valid_index(const int node_index) const {
    return node_index >= 0 && static_cast<std::size_t>(node_index) < entries_.size();
}

void NodeArena::clear() {
    entries_.clear();
    decision_nodes_.clear();
    chance_nodes_.clear();
}

std::size_t NodeArena::size() const {
    return entries_.size();
}

NodeType NodeArena::node_type(const int node_index) const {
    return entry_or_throw(node_index).type;
}

Node& NodeArena::node(const int node_index) {
    const Entry& e = entry_or_throw(node_index);
    if (e.type == NodeType::kDecision) {
        return decision_nodes_.at(static_cast<std::size_t>(e.local_index));
    }
    return chance_nodes_.at(static_cast<std::size_t>(e.local_index));
}

const Node& NodeArena::node(const int node_index) const {
    const Entry& e = entry_or_throw(node_index);
    if (e.type == NodeType::kDecision) {
        return decision_nodes_.at(static_cast<std::size_t>(e.local_index));
    }
    return chance_nodes_.at(static_cast<std::size_t>(e.local_index));
}

DecisionNode& NodeArena::decision(const int node_index) {
    const Entry& e = entry_or_throw(node_index);
    if (e.type != NodeType::kDecision) {
        throw std::invalid_argument("Requested node is not a DecisionNode.");
    }
    return decision_nodes_.at(static_cast<std::size_t>(e.local_index));
}

const DecisionNode& NodeArena::decision(const int node_index) const {
    const Entry& e = entry_or_throw(node_index);
    if (e.type != NodeType::kDecision) {
        throw std::invalid_argument("Requested node is not a DecisionNode.");
    }
    return decision_nodes_.at(static_cast<std::size_t>(e.local_index));
}

ChanceNode& NodeArena::chance(const int node_index) {
    const Entry& e = entry_or_throw(node_index);
    if (e.type != NodeType::kChance) {
        throw std::invalid_argument("Requested node is not a ChanceNode.");
    }
    return chance_nodes_.at(static_cast<std::size_t>(e.local_index));
}

const ChanceNode& NodeArena::chance(const int node_index) const {
    const Entry& e = entry_or_throw(node_index);
    if (e.type != NodeType::kChance) {
        throw std::invalid_argument("Requested node is not a ChanceNode.");
    }
    return chance_nodes_.at(static_cast<std::size_t>(e.local_index));
}

std::vector<int> NodeArena::all_indices() const {
    std::vector<int> indices(entries_.size());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

int NodeArena::append_entry(const NodeType type, const int local_index) {
    entries_.push_back(Entry{type, local_index});
    return static_cast<int>(entries_.size()) - 1;
}

const NodeArena::Entry& NodeArena::entry_or_throw(const int node_index) const {
    if (!valid_index(node_index)) {
        throw std::out_of_range("Node index is out of bounds.");
    }
    return entries_.at(static_cast<std::size_t>(node_index));
}

}  // namespace search
