#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backprop.hpp"
#include "min_max_stats.hpp"
#include "nodes.hpp"
#include "scoring.hpp"
#include "selection.hpp"

#include <random>
#include <string>

using rainbow::search::AverageDiscountedReturnBackpropagator;
using rainbow::search::BackpropConfig;
using rainbow::search::BackpropMethodType;
using rainbow::search::ChanceNode;
using rainbow::search::DecisionNode;
using rainbow::search::MinMaxStats;
using rainbow::search::MinimaxBackpropagator;
using rainbow::search::Node;
using rainbow::search::NodeArena;
using rainbow::search::NodeType;
using rainbow::search::ScoringConfig;
using rainbow::search::ScoringMethodType;
using rainbow::search::SelectionConfig;
using rainbow::search::SelectionMethodType;
namespace py = pybind11;

namespace {

py::dict node_to_dict(const Node& node) {
    py::dict out;
    out["index"] = node.index();
    out["parent_index"] = node.parent_index();
    out["node_type"] = node.is_decision() ? "decision" : "chance";
    out["prior"] = node.prior();
    out["visits"] = node.visits();
    out["value_sum"] = node.value_sum();
    out["to_play"] = node.to_play();
    out["children"] = node.children();
    out["child_priors"] = node.child_priors();
    out["child_values"] = node.child_values();
    out["child_visits"] = node.child_visits();
    return out;
}

std::mt19937_64 build_rng(const SelectionConfig& config) {
    std::random_device rd;
    const uint64_t seed = config.seed == 0 ? (static_cast<uint64_t>(rd()) << 32U) ^ rd() : config.seed;
    return std::mt19937_64(seed);
}

}  // namespace

PYBIND11_MODULE(rainbow_search_cpp, m) {
    m.doc() = "C++ search backend for Rainbow MCTS";

    py::enum_<NodeType>(m, "NodeType")
        .value("DECISION", NodeType::kDecision)
        .value("CHANCE", NodeType::kChance)
        .export_values();

    py::enum_<ScoringMethodType>(m, "ScoringMethodType")
        .value("UCB", ScoringMethodType::kUcb)
        .value("GUMBEL", ScoringMethodType::kGumbel)
        .value("LEAST_VISITED", ScoringMethodType::kLeastVisited)
        .value("PRIOR", ScoringMethodType::kPrior)
        .value("Q_VALUE", ScoringMethodType::kQValue)
        .export_values();

    py::enum_<SelectionMethodType>(m, "SelectionMethodType")
        .value("TOP_SCORE", SelectionMethodType::kTopScore)
        .value("SOFTMAX_SAMPLE", SelectionMethodType::kSoftmaxSample)
        .value("PROBABILITY_SAMPLE", SelectionMethodType::kProbabilitySample)
        .value("MAX_VISIT", SelectionMethodType::kMaxVisit)
        .export_values();

    py::enum_<BackpropMethodType>(m, "BackpropMethodType")
        .value("AVERAGE_DISCOUNTED_RETURN", BackpropMethodType::kAverageDiscountedReturn)
        .value("MINIMAX", BackpropMethodType::kMinimax)
        .export_values();

    py::class_<ScoringConfig>(m, "ScoringConfig")
        .def(py::init<>())
        .def_readwrite("pb_c_init", &ScoringConfig::pb_c_init)
        .def_readwrite("pb_c_base", &ScoringConfig::pb_c_base)
        .def_readwrite("unvisited_value_bootstrap", &ScoringConfig::unvisited_value_bootstrap);

    py::class_<SelectionConfig>(m, "SelectionConfig")
        .def(py::init<>())
        .def_readwrite("temperature", &SelectionConfig::temperature)
        .def_readwrite("random_tiebreak", &SelectionConfig::random_tiebreak)
        .def_readwrite("seed", &SelectionConfig::seed)
        .def_readwrite("mask_value", &SelectionConfig::mask_value);

    py::class_<BackpropConfig>(m, "BackpropConfig")
        .def(py::init<>())
        .def_readwrite("num_players", &BackpropConfig::num_players)
        .def_readwrite("discount_factor", &BackpropConfig::discount_factor)
        .def_readwrite("alternating_minimax", &BackpropConfig::alternating_minimax)
        .def_readwrite("perspective_player", &BackpropConfig::perspective_player);

    py::class_<MinMaxStats>(m, "MinMaxStats")
        .def(
            py::init<const std::vector<double>&, bool, double>(),
            py::arg("known_bounds") = std::vector<double>{},
            py::arg("soft_update") = false,
            py::arg("min_max_epsilon") = 0.01)
        .def("update", &MinMaxStats::update, py::arg("value"))
        .def("normalize", &MinMaxStats::normalize, py::arg("value"))
        .def_property_readonly("min", &MinMaxStats::min)
        .def_property_readonly("max", &MinMaxStats::max)
        .def_property_readonly("soft_update", &MinMaxStats::soft_update)
        .def_property_readonly("min_max_epsilon", &MinMaxStats::min_max_epsilon)
        .def("__repr__", [](const MinMaxStats& stats) { return stats.repr(); });

    py::class_<Node>(m, "Node")
        .def_property("index", &Node::index, &Node::set_index)
        .def_property("parent_index", &Node::parent_index, &Node::set_parent_index)
        .def_property("visits", &Node::visits, &Node::set_visits)
        .def_property("prior", &Node::prior, &Node::set_prior)
        .def_property("value_sum", &Node::value_sum, &Node::set_value_sum)
        .def_property("to_play", &Node::to_play, &Node::set_to_play)
        .def_property_readonly("node_type", &Node::node_type)
        .def_property_readonly("is_decision", &Node::is_decision)
        .def_property_readonly("is_chance", &Node::is_chance)
        .def("value", &Node::value, py::arg("bootstrap") = 0.0)
        .def("expanded", &Node::expanded)
        .def("set_child_stats_size", &Node::set_child_stats_size, py::arg("size"))
        .def("has_child", &Node::has_child, py::arg("key"))
        .def("child_index", &Node::child_index, py::arg("key"))
        .def("set_child", &Node::set_child, py::arg("key"), py::arg("child_index"))
        .def("remove_child", &Node::remove_child, py::arg("key"))
        .def("clear_children", &Node::clear_children)
        .def_property_readonly("children", &Node::children)
        .def_property_readonly("child_keys", &Node::child_keys)
        .def_property_readonly("child_priors", &Node::child_priors)
        .def_property_readonly("child_values", &Node::child_values)
        .def_property_readonly("child_visits", &Node::child_visits)
        .def("__repr__", [](const Node& node) {
            return "<Node index=" + std::to_string(node.index()) +
                   " type=" + (node.is_decision() ? std::string("decision") : std::string("chance")) +
                   " visits=" + std::to_string(node.visits()) + ">";
        });

    py::class_<DecisionNode, Node>(m, "DecisionNode")
        .def(py::init<double, int>(), py::arg("prior") = 0.0, py::arg("parent_index") = -1)
        .def(
            "expand",
            &DecisionNode::expand,
            py::arg("to_play"),
            py::arg("network_policy"),
            py::arg("priors") = std::vector<double>{},
            py::arg("allowed_actions") = std::vector<int>{},
            py::arg("reward") = 0.0,
            py::arg("network_value") = 0.0)
        .def_property("reward", &DecisionNode::reward, &DecisionNode::set_reward)
        .def_property("network_value", &DecisionNode::network_value, &DecisionNode::set_network_value)
        .def_property("stochastic", &DecisionNode::stochastic, &DecisionNode::set_stochastic)
        .def("has_v_mix_cache", &DecisionNode::has_v_mix_cache)
        .def("v_mix_cache", &DecisionNode::v_mix_cache)
        .def("set_v_mix_cache", &DecisionNode::set_v_mix_cache, py::arg("value"))
        .def("clear_v_mix_cache", &DecisionNode::clear_v_mix_cache);

    py::class_<ChanceNode, Node>(m, "ChanceNode")
        .def(py::init<double, int>(), py::arg("prior") = 0.0, py::arg("parent_index") = -1)
        .def(
            "expand",
            &ChanceNode::expand,
            py::arg("to_play"),
            py::arg("network_value"),
            py::arg("code_probs"))
        .def_property("network_value", &ChanceNode::network_value, &ChanceNode::set_network_value)
        .def_property_readonly("code_probs", &ChanceNode::code_probs)
        .def("code_probability", &ChanceNode::code_probability, py::arg("code"))
        .def("set_code_probability", &ChanceNode::set_code_probability, py::arg("code"), py::arg("prob"))
        .def("clear_code_probabilities", &ChanceNode::clear_code_probabilities);

    py::class_<NodeArena>(m, "NodeArena")
        .def(py::init<>())
        .def("create_decision", &NodeArena::create_decision, py::arg("prior") = 0.0, py::arg("parent_index") = -1)
        .def("create_chance", &NodeArena::create_chance, py::arg("prior") = 0.0, py::arg("parent_index") = -1)
        .def("valid_index", &NodeArena::valid_index, py::arg("node_index"))
        .def("clear", &NodeArena::clear)
        .def_property_readonly("size", &NodeArena::size)
        .def("node_type", &NodeArena::node_type, py::arg("node_index"))
        .def("all_indices", &NodeArena::all_indices)
        .def(
            "node",
            [](NodeArena& arena, const int node_index) -> Node& { return arena.node(node_index); },
            py::arg("node_index"),
            py::return_value_policy::reference_internal)
        .def(
            "decision",
            [](NodeArena& arena, const int node_index) -> DecisionNode& { return arena.decision(node_index); },
            py::arg("node_index"),
            py::return_value_policy::reference_internal)
        .def(
            "chance",
            [](NodeArena& arena, const int node_index) -> ChanceNode& { return arena.chance(node_index); },
            py::arg("node_index"),
            py::return_value_policy::reference_internal)
        .def("node_to_dict", [](const NodeArena& arena, const int node_index) {
            return node_to_dict(arena.node(node_index));
        });

    py::class_<AverageDiscountedReturnBackpropagator>(m, "AverageDiscountedReturnBackpropagator")
        .def(py::init<>())
        .def(
            "backpropagate",
            [](const AverageDiscountedReturnBackpropagator& bp,
               NodeArena& arena,
               const std::vector<int>& search_path,
               const std::vector<int>& action_path,
               const double leaf_value,
               const int leaf_to_play,
               MinMaxStats& min_max_stats,
               const BackpropConfig& config) {
                bp.backpropagate(
                    arena,
                    search_path,
                    action_path,
                    leaf_value,
                    leaf_to_play,
                    min_max_stats,
                    config);
            },
            py::arg("arena"),
            py::arg("search_path"),
            py::arg("action_path"),
            py::arg("leaf_value"),
            py::arg("leaf_to_play"),
            py::arg("min_max_stats"),
            py::arg("config"));

    py::class_<MinimaxBackpropagator>(m, "MinimaxBackpropagator")
        .def(py::init<>())
        .def(
            "backpropagate",
            [](const MinimaxBackpropagator& bp,
               NodeArena& arena,
               const std::vector<int>& search_path,
               const std::vector<int>& action_path,
               const double leaf_value,
               const int leaf_to_play,
               MinMaxStats& min_max_stats,
               const BackpropConfig& config) {
                bp.backpropagate(
                    arena,
                    search_path,
                    action_path,
                    leaf_value,
                    leaf_to_play,
                    min_max_stats,
                    config);
            },
            py::arg("arena"),
            py::arg("search_path"),
            py::arg("action_path"),
            py::arg("leaf_value"),
            py::arg("leaf_to_play"),
            py::arg("min_max_stats"),
            py::arg("config"));

    m.def("score_initial", &rainbow::search::score_initial, py::arg("type"), py::arg("prior"), py::arg("action"));
    m.def(
        "compute_scores",
        &rainbow::search::compute_scores,
        py::arg("type"),
        py::arg("arena"),
        py::arg("node_index"),
        py::arg("min_max_stats"),
        py::arg("config"));
    m.def(
        "compute_gumbel_scores_with_policy",
        &rainbow::search::compute_gumbel_scores_with_policy,
        py::arg("arena"),
        py::arg("node_index"),
        py::arg("improved_policy"),
        py::arg("config"));

    m.def("mask_actions", &rainbow::search::mask_actions, py::arg("values"), py::arg("legal_moves"), py::arg("mask_value") = -1e30);
    m.def(
        "select_top_score",
        [](const std::vector<double>& scores, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::select_top_score(scores, config, rng);
        },
        py::arg("scores"),
        py::arg("config"));
    m.def(
        "select_top_score_with_tiebreak",
        [](const std::vector<double>& scores, const std::vector<double>& tiebreak_scores, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::select_top_score_with_tiebreak(scores, tiebreak_scores, config, rng);
        },
        py::arg("scores"),
        py::arg("tiebreak_scores"),
        py::arg("config"));
    m.def(
        "sample_from_softmax",
        [](const std::vector<double>& logits, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::sample_from_softmax(logits, config, rng);
        },
        py::arg("logits"),
        py::arg("config"));
    m.def(
        "sample_from_probabilities",
        [](const std::vector<double>& probs, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::sample_from_probabilities(probs, config, rng);
        },
        py::arg("probs"),
        py::arg("config"));
    m.def(
        "select_max_visit_count",
        [](const NodeArena& arena, const int node_index, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::select_max_visit_count(arena.node(node_index), config, rng);
        },
        py::arg("arena"),
        py::arg("node_index"),
        py::arg("config"));
    m.def(
        "select_chance_outcome",
        [](const NodeArena& arena, const int node_index, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::select_chance_outcome(arena.chance(node_index), config, rng);
        },
        py::arg("arena"),
        py::arg("node_index"),
        py::arg("config"));
    m.def(
        "select_action",
        [](const SelectionMethodType type, const std::vector<double>& scores, const SelectionConfig& config) {
            auto rng = build_rng(config);
            return rainbow::search::select_action(type, scores, config, rng);
        },
        py::arg("type"),
        py::arg("scores"),
        py::arg("config"));

    m.def(
        "compute_child_q_from_parent",
        &rainbow::search::compute_child_q_from_parent,
        py::arg("arena"),
        py::arg("parent_index"),
        py::arg("child_index"),
        py::arg("config"));
    m.def(
        "backpropagate_with_method",
        &rainbow::search::backpropagate_with_method,
        py::arg("type"),
        py::arg("arena"),
        py::arg("search_path"),
        py::arg("action_path"),
        py::arg("leaf_value"),
        py::arg("leaf_to_play"),
        py::arg("min_max_stats"),
        py::arg("config"));
}
