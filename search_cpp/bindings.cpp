#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "min_max_stats.hpp"
#include "nodes.hpp"

#include <string>

namespace py = pybind11;
using rainbow::search::ChanceNode;
using rainbow::search::DecisionNode;
using rainbow::search::MinMaxStats;
using rainbow::search::Node;
using rainbow::search::NodeArena;
using rainbow::search::NodeType;

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

}  // namespace

PYBIND11_MODULE(rainbow_search_cpp, m) {
    m.doc() = "C++ search backend for Rainbow MCTS";

    py::enum_<NodeType>(m, "NodeType")
        .value("DECISION", NodeType::kDecision)
        .value("CHANCE", NodeType::kChance)
        .export_values();

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
}
