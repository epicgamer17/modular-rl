#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backprop.hpp"
#include "min_max_stats.hpp"
#include "modular_search.hpp"
#include "nodes.hpp"
#include "scoring.hpp"
#include "selection.hpp"

#include <random>
#include <string>

using search::AverageDiscountedReturnBackpropagator;
using search::BackpropConfig;
using search::BackpropMethodType;
using search::ChanceNode;
using search::DecisionNode;
using search::HiddenInferenceUpdateBatch;
using search::LeafBatchRequest;
using search::MinimaxBackpropagator;
using search::MinMaxStats;
using search::Node;
using search::NodeArena;
using search::NodeType;
using search::ScoringConfig;
using search::ScoringMethodType;
using search::SearchAlgorithm;
using search::SearchConfig;
using search::SelectionConfig;
using search::SelectionMethodType;
namespace py = pybind11;

namespace {

py::dict node_to_dict(const Node &node) {
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

std::mt19937_64 build_rng(const SelectionConfig &config) {
  std::random_device rd;
  const uint64_t seed = config.seed == 0
                            ? (static_cast<uint64_t>(rd()) << 32U) ^ rd()
                            : config.seed;
  return std::mt19937_64(seed);
}

template <typename T>
py::array_t<T> vector_to_array_1d(const std::vector<T> &values) {
  py::array_t<T> out(values.size());
  auto view = out.template mutable_unchecked<1>();
  for (ssize_t i = 0; i < static_cast<ssize_t>(values.size()); ++i) {
    view(i) = values[static_cast<std::size_t>(i)];
  }
  return out;
}

template <typename T>
std::vector<T> array_to_vector_1d(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &arr) {
  std::vector<T> out(static_cast<std::size_t>(arr.size()));
  auto view = arr.template unchecked<1>();
  for (ssize_t i = 0; i < view.shape(0); ++i) {
    out[static_cast<std::size_t>(i)] = view(i);
  }
  return out;
}

template <typename T>
std::vector<T> array_to_vector_2d_flat(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &arr,
    int &rows, int &cols) {
  auto view = arr.template unchecked<2>();
  rows = static_cast<int>(view.shape(0));
  cols = static_cast<int>(view.shape(1));
  std::vector<T> out(static_cast<std::size_t>(rows) *
                     static_cast<std::size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
          static_cast<std::size_t>(c)] = view(r, c);
    }
  }
  return out;
}

py::dict leaf_batch_request_to_dict(const LeafBatchRequest &req) {
  py::dict out;
  out["hidden_request_ids"] =
      vector_to_array_1d<int32_t>(req.hidden_request_ids);
  out["hidden_parent_state_handles"] =
      vector_to_array_1d<int64_t>(req.hidden_parent_state_handles);
  out["hidden_actions"] = vector_to_array_1d<int32_t>(req.hidden_actions);
  out["hidden_action_is_one_hot"] =
      vector_to_array_1d<uint8_t>(req.hidden_action_is_one_hot);
  out["hidden_num_codes"] = vector_to_array_1d<int32_t>(req.hidden_num_codes);
  out["afterstate_request_ids"] =
      vector_to_array_1d<int32_t>(req.afterstate_request_ids);
  out["afterstate_parent_state_handles"] =
      vector_to_array_1d<int64_t>(req.afterstate_parent_state_handles);
  out["afterstate_actions"] =
      vector_to_array_1d<int32_t>(req.afterstate_actions);
  return out;
}

} // namespace

PYBIND11_MODULE(mcts_cpp_backend, m) {
  m.doc() = "C++ search backend for Tree Search";

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
      .value("AVERAGE_DISCOUNTED_RETURN",
             BackpropMethodType::kAverageDiscountedReturn)
      .value("MINIMAX", BackpropMethodType::kMinimax)
      .export_values();

  py::class_<SearchConfig>(m, "SearchConfig")
      .def(py::init<>())
      .def_readwrite("num_actions", &SearchConfig::num_actions)
      .def_readwrite("num_players", &SearchConfig::num_players)
      .def_readwrite("default_batch_size", &SearchConfig::default_batch_size)
      .def_readwrite("stochastic", &SearchConfig::stochastic)
      .def_readwrite("known_bounds", &SearchConfig::known_bounds)
      .def_readwrite("soft_update_minmax", &SearchConfig::soft_update_minmax)
      .def_readwrite("min_max_epsilon", &SearchConfig::min_max_epsilon)
      .def_readwrite("discount_factor", &SearchConfig::discount_factor)
      .def_readwrite("alternating_minimax", &SearchConfig::alternating_minimax)
      .def_readwrite("perspective_player", &SearchConfig::perspective_player);

  py::class_<ScoringConfig>(m, "ScoringConfig")
      .def(py::init<>())
      .def_readwrite("pb_c_init", &ScoringConfig::pb_c_init)
      .def_readwrite("pb_c_base", &ScoringConfig::pb_c_base)
      .def_readwrite("unvisited_value_bootstrap",
                     &ScoringConfig::unvisited_value_bootstrap);

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
      .def_readwrite("alternating_minimax",
                     &BackpropConfig::alternating_minimax)
      .def_readwrite("perspective_player", &BackpropConfig::perspective_player);

  py::class_<MinMaxStats>(m, "MinMaxStats")
      .def(py::init<const std::vector<double> &, bool, double>(),
           py::arg("known_bounds") = std::vector<double>{},
           py::arg("soft_update") = false, py::arg("min_max_epsilon") = 0.01)
      .def("update", &MinMaxStats::update, py::arg("value"))
      .def("normalize", &MinMaxStats::normalize, py::arg("value"))
      .def_property_readonly("min", &MinMaxStats::min)
      .def_property_readonly("max", &MinMaxStats::max)
      .def_property_readonly("soft_update", &MinMaxStats::soft_update)
      .def_property_readonly("min_max_epsilon", &MinMaxStats::min_max_epsilon)
      .def("__repr__", [](const MinMaxStats &stats) { return stats.repr(); });

  py::class_<Node>(m, "Node")
      .def_property("index", &Node::index, &Node::set_index)
      .def_property("parent_index", &Node::parent_index,
                    &Node::set_parent_index)
      .def_property("visits", &Node::visits, &Node::set_visits)
      .def_property("prior", &Node::prior, &Node::set_prior)
      .def_property("value_sum", &Node::value_sum, &Node::set_value_sum)
      .def_property("to_play", &Node::to_play, &Node::set_to_play)
      .def_property("state_handle", &Node::state_handle,
                    &Node::set_state_handle)
      .def_property_readonly("node_type", &Node::node_type)
      .def_property_readonly("is_decision", &Node::is_decision)
      .def_property_readonly("is_chance", &Node::is_chance)
      .def("value", &Node::value, py::arg("bootstrap") = 0.0)
      .def("expanded", &Node::expanded)
      .def("set_child_stats_size", &Node::set_child_stats_size, py::arg("size"))
      .def("has_child", &Node::has_child, py::arg("key"))
      .def("child_index", &Node::child_index, py::arg("key"))
      .def("set_child", &Node::set_child, py::arg("key"),
           py::arg("child_index"))
      .def("remove_child", &Node::remove_child, py::arg("key"))
      .def("clear_children", &Node::clear_children)
      .def_property_readonly("children", &Node::children)
      .def_property_readonly("child_keys", &Node::child_keys)
      .def_property_readonly("child_priors", &Node::child_priors)
      .def_property_readonly("child_values", &Node::child_values)
      .def_property_readonly("child_visits", &Node::child_visits)
      .def("__repr__", [](const Node &node) {
        return "<Node index=" + std::to_string(node.index()) + " type=" +
               (node.is_decision() ? std::string("decision")
                                   : std::string("chance")) +
               " visits=" + std::to_string(node.visits()) + ">";
      });

  py::class_<DecisionNode, Node>(m, "DecisionNode")
      .def(py::init<double, int>(), py::arg("prior") = 0.0,
           py::arg("parent_index") = -1)
      .def("expand", &DecisionNode::expand, py::arg("to_play"),
           py::arg("network_policy"), py::arg("priors") = std::vector<double>{},
           py::arg("allowed_actions") = std::vector<int>{},
           py::arg("reward") = 0.0, py::arg("network_value") = 0.0)
      .def_property("reward", &DecisionNode::reward, &DecisionNode::set_reward)
      .def_property("network_value", &DecisionNode::network_value,
                    &DecisionNode::set_network_value)
      .def_property("stochastic", &DecisionNode::stochastic,
                    &DecisionNode::set_stochastic)
      .def("has_v_mix_cache", &DecisionNode::has_v_mix_cache)
      .def("v_mix_cache", &DecisionNode::v_mix_cache)
      .def("set_v_mix_cache", &DecisionNode::set_v_mix_cache, py::arg("value"))
      .def("clear_v_mix_cache", &DecisionNode::clear_v_mix_cache);

  py::class_<ChanceNode, Node>(m, "ChanceNode")
      .def(py::init<double, int>(), py::arg("prior") = 0.0,
           py::arg("parent_index") = -1)
      .def("expand", &ChanceNode::expand, py::arg("to_play"),
           py::arg("network_value"), py::arg("code_probs"))
      .def_property("network_value", &ChanceNode::network_value,
                    &ChanceNode::set_network_value)
      .def_property_readonly("code_probs", &ChanceNode::code_probs)
      .def("code_probability", &ChanceNode::code_probability, py::arg("code"))
      .def("set_code_probability", &ChanceNode::set_code_probability,
           py::arg("code"), py::arg("prob"))
      .def("clear_code_probabilities", &ChanceNode::clear_code_probabilities);

  py::class_<NodeArena>(m, "NodeArena")
      .def(py::init<>())
      .def("create_decision", &NodeArena::create_decision,
           py::arg("prior") = 0.0, py::arg("parent_index") = -1)
      .def("create_chance", &NodeArena::create_chance, py::arg("prior") = 0.0,
           py::arg("parent_index") = -1)
      .def("valid_index", &NodeArena::valid_index, py::arg("node_index"))
      .def("clear", &NodeArena::clear)
      .def_property_readonly("size", &NodeArena::size)
      .def("node_type", &NodeArena::node_type, py::arg("node_index"))
      .def("all_indices", &NodeArena::all_indices)
      .def(
          "node",
          [](NodeArena &arena, const int node_index) -> Node & {
            return arena.node(node_index);
          },
          py::arg("node_index"), py::return_value_policy::reference_internal)
      .def(
          "decision",
          [](NodeArena &arena, const int node_index) -> DecisionNode & {
            return arena.decision(node_index);
          },
          py::arg("node_index"), py::return_value_policy::reference_internal)
      .def(
          "chance",
          [](NodeArena &arena, const int node_index) -> ChanceNode & {
            return arena.chance(node_index);
          },
          py::arg("node_index"), py::return_value_policy::reference_internal)
      .def("node_to_dict", [](const NodeArena &arena, const int node_index) {
        return node_to_dict(arena.node(node_index));
      });

  py::class_<SearchAlgorithm>(m, "SearchAlgorithm")
      .def(py::init<const SearchConfig &, const ScoringConfig &,
                    const SelectionConfig &, const SelectionConfig &,
                    const SelectionConfig &, const BackpropConfig &,
                    ScoringMethodType, ScoringMethodType, SelectionMethodType,
                    SelectionMethodType, SelectionMethodType,
                    BackpropMethodType>(),
           py::arg("search_config"),
           py::arg("scoring_config") = ScoringConfig{},
           py::arg("root_selection_config") = SelectionConfig{},
           py::arg("decision_selection_config") = SelectionConfig{},
           py::arg("chance_selection_config") = SelectionConfig{},
           py::arg("backprop_config") = BackpropConfig{},
           py::arg("root_scoring_type") = ScoringMethodType::kUcb,
           py::arg("decision_scoring_type") = ScoringMethodType::kUcb,
           py::arg("root_selection_type") = SelectionMethodType::kTopScore,
           py::arg("decision_selection_type") = SelectionMethodType::kTopScore,
           py::arg("chance_selection_type") =
               SelectionMethodType::kProbabilitySample,
           py::arg("backprop_method") =
               BackpropMethodType::kAverageDiscountedReturn)
      .def("clear", &SearchAlgorithm::clear)
      .def_property_readonly("has_root", &SearchAlgorithm::has_root)
      .def_property_readonly("root_index", &SearchAlgorithm::root_index)
      .def_property_readonly("node_count", &SearchAlgorithm::node_count)
      .def_property_readonly("pending_count", &SearchAlgorithm::pending_count)
      .def("initialize_root", &SearchAlgorithm::initialize_root,
           py::arg("policy_priors"), py::arg("to_play"),
           py::arg("state_handle"), py::arg("root_value") = 0.0,
           py::arg("allowed_actions") = std::vector<int>{})
      .def(
          "step_search_until_leaves",
          [](SearchAlgorithm &search, int batch_size) {
            LeafBatchRequest req;
            {
              py::gil_scoped_release release;
              req = search.step_search_until_leaves(batch_size);
            }
            return leaf_batch_request_to_dict(req);
          },
          py::arg("batch_size") = -1)
      .def(
          "update_leaves_and_backprop",
          [](SearchAlgorithm &search,
             const py::array_t<int32_t,
                               py::array::c_style | py::array::forcecast>
                 &hidden_request_ids,
             const py::array_t<int64_t,
                               py::array::c_style | py::array::forcecast>
                 &hidden_next_state_handles,
             const py::array_t<double,
                               py::array::c_style | py::array::forcecast>
                 &hidden_rewards,
             const py::array_t<double, py::array::c_style |
                                           py::array::forcecast> &hidden_values,
             const py::array_t<int32_t,
                               py::array::c_style | py::array::forcecast>
                 &hidden_to_plays,
             const py::array_t<double, py::array::c_style |
                                           py::array::forcecast> &hidden_priors,
             const py::array_t<int32_t,
                               py::array::c_style | py::array::forcecast>
                 &afterstate_request_ids,
             const py::array_t<int64_t,
                               py::array::c_style | py::array::forcecast>
                 &afterstate_next_state_handles,
             const py::array_t<double,
                               py::array::c_style | py::array::forcecast>
                 &afterstate_values,
             const py::array_t<double,
                               py::array::c_style | py::array::forcecast>
                 &afterstate_code_probs) {
            HiddenInferenceUpdateBatch hidden_updates;
            hidden_updates.request_ids =
                array_to_vector_1d<int32_t>(hidden_request_ids);
            hidden_updates.next_state_handles =
                array_to_vector_1d<int64_t>(hidden_next_state_handles);
            hidden_updates.rewards = array_to_vector_1d<double>(hidden_rewards);
            hidden_updates.values = array_to_vector_1d<double>(hidden_values);
            hidden_updates.to_plays =
                array_to_vector_1d<int32_t>(hidden_to_plays);
            int hidden_rows = 0;
            int hidden_cols = 0;
            hidden_updates.priors = array_to_vector_2d_flat<double>(
                hidden_priors, hidden_rows, hidden_cols);
            hidden_updates.num_actions = hidden_cols;

            HiddenInferenceUpdateBatch empty_hidden;
            if (hidden_rows == 0 && hidden_updates.request_ids.empty()) {
              hidden_updates = std::move(empty_hidden);
            }

            search::AfterstateInferenceUpdateBatch after_updates;
            after_updates.request_ids =
                array_to_vector_1d<int32_t>(afterstate_request_ids);
            after_updates.next_state_handles =
                array_to_vector_1d<int64_t>(afterstate_next_state_handles);
            after_updates.values =
                array_to_vector_1d<double>(afterstate_values);
            int after_rows = 0;
            int after_cols = 0;
            after_updates.code_probs = array_to_vector_2d_flat<double>(
                afterstate_code_probs, after_rows, after_cols);
            after_updates.num_codes = after_cols;

            search::AfterstateInferenceUpdateBatch empty_after;
            if (after_rows == 0 && after_updates.request_ids.empty()) {
              after_updates = std::move(empty_after);
            }

            int processed = 0;
            {
              py::gil_scoped_release release;
              processed = search.update_leaves_and_backprop(hidden_updates,
                                                            after_updates);
            }
            return processed;
          },
          py::arg("hidden_request_ids"), py::arg("hidden_next_state_handles"),
          py::arg("hidden_rewards"), py::arg("hidden_values"),
          py::arg("hidden_to_plays"), py::arg("hidden_priors"),
          py::arg("afterstate_request_ids"),
          py::arg("afterstate_next_state_handles"),
          py::arg("afterstate_values"), py::arg("afterstate_code_probs"))
      .def("root_value", &SearchAlgorithm::root_value)
      .def("root_child_priors", &SearchAlgorithm::root_child_priors)
      .def("root_child_values", &SearchAlgorithm::root_child_values)
      .def("root_child_visits", &SearchAlgorithm::root_child_visits)
      .def("select_root_action", &SearchAlgorithm::select_root_action,
           py::arg("method") = SelectionMethodType::kMaxVisit);

  py::class_<AverageDiscountedReturnBackpropagator>(
      m, "AverageDiscountedReturnBackpropagator")
      .def(py::init<>())
      .def(
          "backpropagate",
          [](const AverageDiscountedReturnBackpropagator &bp, NodeArena &arena,
             const std::vector<int> &search_path,
             const std::vector<int> &action_path, const double leaf_value,
             const int leaf_to_play, MinMaxStats &min_max_stats,
             const BackpropConfig &config) {
            bp.backpropagate(arena, search_path, action_path, leaf_value,
                             leaf_to_play, min_max_stats, config);
          },
          py::arg("arena"), py::arg("search_path"), py::arg("action_path"),
          py::arg("leaf_value"), py::arg("leaf_to_play"),
          py::arg("min_max_stats"), py::arg("config"));

  py::class_<MinimaxBackpropagator>(m, "MinimaxBackpropagator")
      .def(py::init<>())
      .def(
          "backpropagate",
          [](const MinimaxBackpropagator &bp, NodeArena &arena,
             const std::vector<int> &search_path,
             const std::vector<int> &action_path, const double leaf_value,
             const int leaf_to_play, MinMaxStats &min_max_stats,
             const BackpropConfig &config) {
            bp.backpropagate(arena, search_path, action_path, leaf_value,
                             leaf_to_play, min_max_stats, config);
          },
          py::arg("arena"), py::arg("search_path"), py::arg("action_path"),
          py::arg("leaf_value"), py::arg("leaf_to_play"),
          py::arg("min_max_stats"), py::arg("config"));

  m.def("score_initial", &search::score_initial, py::arg("type"),
        py::arg("prior"), py::arg("action"));
  m.def("compute_scores", &search::compute_scores, py::arg("type"),
        py::arg("arena"), py::arg("node_index"), py::arg("min_max_stats"),
        py::arg("config"));
  m.def("compute_gumbel_scores_with_policy",
        &search::compute_gumbel_scores_with_policy, py::arg("arena"),
        py::arg("node_index"), py::arg("improved_policy"), py::arg("config"));

  m.def("mask_actions", &search::mask_actions, py::arg("values"),
        py::arg("legal_moves"), py::arg("mask_value") = -1e30);
  m.def(
      "select_top_score",
      [](const std::vector<double> &scores, const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::select_top_score(scores, config, rng);
      },
      py::arg("scores"), py::arg("config"));
  m.def(
      "select_top_score_with_tiebreak",
      [](const std::vector<double> &scores,
         const std::vector<double> &tiebreak_scores,
         const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::select_top_score_with_tiebreak(scores, tiebreak_scores,
                                                      config, rng);
      },
      py::arg("scores"), py::arg("tiebreak_scores"), py::arg("config"));
  m.def(
      "sample_from_softmax",
      [](const std::vector<double> &logits, const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::sample_from_softmax(logits, config, rng);
      },
      py::arg("logits"), py::arg("config"));
  m.def(
      "sample_from_probabilities",
      [](const std::vector<double> &probs, const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::sample_from_probabilities(probs, config, rng);
      },
      py::arg("probs"), py::arg("config"));
  m.def(
      "select_max_visit_count",
      [](const NodeArena &arena, const int node_index,
         const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::select_max_visit_count(arena.node(node_index), config,
                                              rng);
      },
      py::arg("arena"), py::arg("node_index"), py::arg("config"));
  m.def(
      "select_chance_outcome",
      [](const NodeArena &arena, const int node_index,
         const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::select_chance_outcome(arena.chance(node_index), config,
                                             rng);
      },
      py::arg("arena"), py::arg("node_index"), py::arg("config"));
  m.def(
      "select_action",
      [](const SelectionMethodType type, const std::vector<double> &scores,
         const SelectionConfig &config) {
        auto rng = build_rng(config);
        return search::select_action(type, scores, config, rng);
      },
      py::arg("type"), py::arg("scores"), py::arg("config"));

  m.def("compute_child_q_from_parent", &search::compute_child_q_from_parent,
        py::arg("arena"), py::arg("parent_index"), py::arg("child_index"),
        py::arg("config"));
  m.def("backpropagate_with_method", &search::backpropagate_with_method,
        py::arg("type"), py::arg("arena"), py::arg("search_path"),
        py::arg("action_path"), py::arg("leaf_value"), py::arg("leaf_to_play"),
        py::arg("min_max_stats"), py::arg("config"));
}
