#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace search {

enum class NodeType {
    kDecision = 0,
    kChance = 1,
};

class Node {
public:
    Node(NodeType node_type, double prior, int parent_index = -1);
    virtual ~Node() = default;

    NodeType node_type() const;
    bool is_decision() const;
    bool is_chance() const;

    int index() const;
    void set_index(int index);

    int parent_index() const;
    void set_parent_index(int parent_index);

    int visits() const;
    void set_visits(int visits);
    void add_visit(int delta = 1);

    double prior() const;
    void set_prior(double prior);

    double value_sum() const;
    void set_value_sum(double value_sum);
    void add_value(double delta);

    int to_play() const;
    void set_to_play(int to_play);

    int64_t state_handle() const;
    void set_state_handle(int64_t state_handle);

    double value(double bootstrap = 0.0) const;
    bool expanded() const;

    void set_child_stats_size(std::size_t size);
    const std::vector<double>& child_priors() const;
    const std::vector<double>& child_values() const;
    const std::vector<double>& child_visits() const;
    std::vector<double>& mutable_child_priors();
    std::vector<double>& mutable_child_values();
    std::vector<double>& mutable_child_visits();

    bool has_child(int key) const;
    int child_index(int key) const;
    void set_child(int key, int child_index);
    void remove_child(int key);
    void clear_children();
    const std::unordered_map<int, int>& children() const;
    std::vector<int> child_keys() const;

private:
    NodeType node_type_;
    int index_;
    int parent_index_;
    int visits_;
    int to_play_;
    int64_t state_handle_;
    double prior_;
    double value_sum_;
    std::unordered_map<int, int> children_;
    std::vector<double> child_priors_;
    std::vector<double> child_values_;
    std::vector<double> child_visits_;
};

class DecisionNode final : public Node {
public:
    explicit DecisionNode(double prior = 0.0, int parent_index = -1);

    void expand(
        int to_play,
        const std::vector<double>& network_policy,
        const std::vector<double>& priors = {},
        const std::vector<int>& allowed_actions = {},
        double reward = 0.0,
        double network_value = 0.0);

    double reward() const;
    void set_reward(double reward);

    double network_value() const;
    void set_network_value(double network_value);

    bool stochastic() const;
    void set_stochastic(bool stochastic);

    bool has_v_mix_cache() const;
    double v_mix_cache() const;
    void set_v_mix_cache(double v_mix_cache);
    void clear_v_mix_cache();

private:
    std::vector<double> network_policy_;
    double reward_;
    double network_value_;
    bool stochastic_;
    bool has_v_mix_cache_;
    double v_mix_cache_;
};

class ChanceNode final : public Node {
public:
    explicit ChanceNode(double prior = 0.0, int parent_index = -1);

    void expand(
        int to_play,
        double network_value,
        const std::vector<double>& code_probs);

    double network_value() const;
    void set_network_value(double network_value);

    const std::unordered_map<int, double>& code_probs() const;
    double code_probability(int code) const;
    void set_code_probability(int code, double prob);
    void clear_code_probabilities();

private:
    double network_value_;
    std::unordered_map<int, double> code_probs_;
};

class NodeArena {
public:
    NodeArena() = default;

    int create_decision(double prior = 0.0, int parent_index = -1);
    int create_chance(double prior = 0.0, int parent_index = -1);

    bool valid_index(int node_index) const;
    void clear();
    std::size_t size() const;

    NodeType node_type(int node_index) const;
    Node& node(int node_index);
    const Node& node(int node_index) const;

    DecisionNode& decision(int node_index);
    const DecisionNode& decision(int node_index) const;
    ChanceNode& chance(int node_index);
    const ChanceNode& chance(int node_index) const;

    std::vector<int> all_indices() const;

private:
    struct Entry {
        NodeType type;
        int local_index;
    };

    int append_entry(NodeType type, int local_index);
    const Entry& entry_or_throw(int node_index) const;

    std::vector<Entry> entries_;
    std::vector<DecisionNode> decision_nodes_;
    std::vector<ChanceNode> chance_nodes_;
};

}  // namespace search
