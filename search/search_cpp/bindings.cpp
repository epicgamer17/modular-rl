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
#include <stdexcept>
#include <string>

using search::AverageDiscountedReturnBackpropagator;
using search::BackpropConfig;
using search::BackpropMethodType;
using search::ChanceNode;
using search::DecisionNode;
using search::LeafBatchRequest;
using search::MinimaxBackpropagator;
using search::MinMaxStats;
using search::Node;
using search::NodeArena;
using search::NodeType;
using search::ScoringConfig;
using search::ScoringMethodType;
using search::ModularSearch;
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
py::array_t<T> vector_to_numpy_view(std::vector<T> &values, py::handle owner) {
  return py::array_t<T>(
      {static_cast<py::ssize_t>(values.size())},
      {static_cast<py::ssize_t>(sizeof(T))},
      values.empty() ? nullptr : values.data(),
      owner);
}

} // namespace

PYBIND11_MODULE(search_cpp, m) {
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
      .def_readwrite("num_simulations", &SearchConfig::num_simulations)
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

  py::class_<LeafBatchRequest>(m, "LeafBatchRequest")
      .def(py::init<>())
      .def_property_readonly("hidden_request_ids",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(req.hidden_request_ids,
                                                           owner);
                             })
      .def_property_readonly("hidden_parent_state_handles",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(
                                   req.hidden_parent_state_handles, owner);
                             })
      .def_property_readonly("hidden_actions",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(req.hidden_actions,
                                                           owner);
                             })
      .def_property_readonly("hidden_action_is_one_hot",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(
                                   req.hidden_action_is_one_hot, owner);
                             })
      .def_property_readonly("hidden_num_codes",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(req.hidden_num_codes,
                                                           owner);
                             })
      .def_property_readonly("afterstate_request_ids",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(
                                   req.afterstate_request_ids, owner);
                             })
      .def_property_readonly("afterstate_parent_state_handles",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(
                                   req.afterstate_parent_state_handles, owner);
                             })
      .def_property_readonly("afterstate_actions",
                             [](LeafBatchRequest &req) {
                               py::object owner = py::cast(
                                   &req, py::return_value_policy::reference_internal);
                               return vector_to_numpy_view(req.afterstate_actions,
                                                           owner);
                             })
      .def_property_readonly("total_size", &LeafBatchRequest::total_size)
      .def_property_readonly("empty", &LeafBatchRequest::empty);

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
      .def("reserve", &NodeArena::reserve, py::arg("capacity"))
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

  py::class_<ModularSearch>(m, "ModularSearch", py::dynamic_attr())
      .def(py::init([](py::object config, py::object device, int num_actions) {
        SearchConfig sc;
        sc.num_actions = num_actions;
        sc.num_simulations = py::getattr(config, "num_simulations").cast<int>();
        sc.discount_factor = py::getattr(config, "discount_factor").cast<double>();
        sc.stochastic = py::getattr(config, "stochastic", py::cast(false)).cast<bool>();
        
        // Extract bounds
        if (py::hasattr(config, "known_bounds") && !py::getattr(config, "known_bounds").is_none()) {
            sc.known_bounds = py::getattr(config, "known_bounds").cast<std::vector<double>>();
        }

        ScoringConfig scoring_cfg;
        scoring_cfg.pb_c_init = py::getattr(config, "pb_c_init").cast<double>();
        scoring_cfg.pb_c_base = py::getattr(config, "pb_c_base").cast<double>();

        SelectionConfig root_selection_cfg;
        root_selection_cfg.temperature = 1.0;
        
        SelectionConfig decision_selection_cfg;
        decision_selection_cfg.temperature = 1.0;

        SelectionConfig chance_selection_cfg;
        chance_selection_cfg.temperature = 1.0;

        BackpropConfig backprop_cfg;
        backprop_cfg.discount_factor = sc.discount_factor;

        ScoringMethodType root_scoring = ScoringMethodType::kUcb;
        ScoringMethodType decision_scoring = ScoringMethodType::kUcb;
        
        if (py::getattr(config, "gumbel", py::cast(false)).cast<bool>()) {
            root_scoring = ScoringMethodType::kGumbel;
            decision_scoring = ScoringMethodType::kGumbel;
        }

        auto self = std::make_unique<ModularSearch>(
            sc, scoring_cfg, root_selection_cfg, decision_selection_cfg,
            chance_selection_cfg, backprop_cfg, root_scoring, decision_scoring,
            SelectionMethodType::kTopScore, SelectionMethodType::kTopScore,
            SelectionMethodType::kProbabilitySample,
            BackpropMethodType::kAverageDiscountedReturn);
        
        // Use set_attr on the Python wrapper to store these
        // Note: 'self' is the pointer, pybind11 will handle the wrapping.
        return self;
      }))
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
      .def("clear", &ModularSearch::clear)
      .def_property_readonly("has_root", &ModularSearch::has_root)
      .def_property_readonly("root_index", &ModularSearch::root_index)
      .def_property_readonly("node_count", &ModularSearch::node_count)
      .def_property_readonly("pending_count", &ModularSearch::pending_count)
      .def("initialize_root", &ModularSearch::initialize_root,
           py::arg("policy_priors"), py::arg("to_play"),
           py::arg("state_handle"), py::arg("root_value") = 0.0,
           py::arg("allowed_actions") = std::vector<int>{})
      .def(
          "step_search_until_leaves",
          [](ModularSearch &search, int batch_size) {
            LeafBatchRequest req;
            {
              py::gil_scoped_release release;
              req = search.step_search_until_leaves(batch_size);
            }
            return req;
          },
          py::arg("batch_size") = -1)
      .def(
          "update_leaves_and_backprop",
          [](ModularSearch &search,
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
            auto hidden_request_ids_view = hidden_request_ids.unchecked<1>();
            auto hidden_next_state_handles_view =
                hidden_next_state_handles.unchecked<1>();
            auto hidden_rewards_view = hidden_rewards.unchecked<1>();
            auto hidden_values_view = hidden_values.unchecked<1>();
            auto hidden_to_plays_view = hidden_to_plays.unchecked<1>();
            auto hidden_priors_view = hidden_priors.unchecked<2>();

            auto afterstate_request_ids_view =
                afterstate_request_ids.unchecked<1>();
            auto afterstate_next_state_handles_view =
                afterstate_next_state_handles.unchecked<1>();
            auto afterstate_values_view = afterstate_values.unchecked<1>();
            auto afterstate_code_probs_view = afterstate_code_probs.unchecked<2>();

            const py::ssize_t hidden_count = hidden_request_ids_view.shape(0);
            const py::ssize_t afterstate_count =
                afterstate_request_ids_view.shape(0);
            const int hidden_num_actions =
                static_cast<int>(hidden_priors_view.shape(1));
            const int afterstate_num_codes =
                static_cast<int>(afterstate_code_probs_view.shape(1));

            if (hidden_next_state_handles_view.shape(0) != hidden_count ||
                hidden_rewards_view.shape(0) != hidden_count ||
                hidden_values_view.shape(0) != hidden_count ||
                hidden_to_plays_view.shape(0) != hidden_count ||
                hidden_priors_view.shape(0) != hidden_count) {
              throw std::invalid_argument(
                  "Hidden update arrays must agree on batch dimension.");
            }
            if (afterstate_next_state_handles_view.shape(0) != afterstate_count ||
                afterstate_values_view.shape(0) != afterstate_count ||
                afterstate_code_probs_view.shape(0) != afterstate_count) {
              throw std::invalid_argument(
                  "Afterstate update arrays must agree on batch dimension.");
            }

            const int32_t *hidden_request_ids_ptr =
                hidden_count > 0 ? &hidden_request_ids_view(0) : nullptr;
            const int64_t *hidden_next_state_handles_ptr =
                hidden_count > 0 ? &hidden_next_state_handles_view(0) : nullptr;
            const double *hidden_rewards_ptr =
                hidden_count > 0 ? &hidden_rewards_view(0) : nullptr;
            const double *hidden_values_ptr =
                hidden_count > 0 ? &hidden_values_view(0) : nullptr;
            const int32_t *hidden_to_plays_ptr =
                hidden_count > 0 ? &hidden_to_plays_view(0) : nullptr;
            const double *hidden_priors_ptr =
                (hidden_count > 0 && hidden_num_actions > 0)
                    ? &hidden_priors_view(0, 0)
                    : nullptr;

            const int32_t *afterstate_request_ids_ptr =
                afterstate_count > 0 ? &afterstate_request_ids_view(0) : nullptr;
            const int64_t *afterstate_next_state_handles_ptr =
                afterstate_count > 0 ? &afterstate_next_state_handles_view(0)
                                     : nullptr;
            const double *afterstate_values_ptr =
                afterstate_count > 0 ? &afterstate_values_view(0) : nullptr;
            const double *afterstate_code_probs_ptr =
                (afterstate_count > 0 && afterstate_num_codes > 0)
                    ? &afterstate_code_probs_view(0, 0)
                    : nullptr;

            int processed = 0;
            {
              py::gil_scoped_release release;
              processed = search.update_leaves_and_backprop_raw(
                  hidden_request_ids_ptr, hidden_next_state_handles_ptr,
                  hidden_rewards_ptr, hidden_values_ptr, hidden_to_plays_ptr,
                  hidden_priors_ptr, static_cast<std::size_t>(hidden_count),
                  hidden_num_actions, afterstate_request_ids_ptr,
                  afterstate_next_state_handles_ptr, afterstate_values_ptr,
                  afterstate_code_probs_ptr,
                  static_cast<std::size_t>(afterstate_count),
                  afterstate_num_codes);
            }
            return processed;
          },
          py::arg("hidden_request_ids"), py::arg("hidden_next_state_handles"),
          py::arg("hidden_rewards"), py::arg("hidden_values"),
          py::arg("hidden_to_plays"), py::arg("hidden_priors"),
          py::arg("afterstate_request_ids"),
          py::arg("afterstate_next_state_handles"),
          py::arg("afterstate_values"), py::arg("afterstate_code_probs"))
      .def("run", [](ModularSearch &self, py::object obs, py::dict info, py::object agent_network, py::object trajectory_actions, bool exploration) {
          // Wrap single-element case into run_vectorized for DRY
          // FIX: Use torch.as_tensor and unsqueeze to keep it as a tensor on the correct device
          py::object torch = py::module_::import("torch");
          py::object batched_obs = torch.attr("as_tensor")(obs);
          if (py::hasattr(agent_network, "input_shape")) {
              auto input_shape = agent_network.attr("input_shape");
              if (batched_obs.attr("dim")().cast<int>() == py::len(input_shape)) {
                  batched_obs = batched_obs.attr("unsqueeze")(0);
              }
          } else {
              py::tuple shape = batched_obs.attr("shape").cast<py::tuple>();
              if (py::len(shape) > 0 && shape[0].cast<int>() != 1) {
                  batched_obs = batched_obs.attr("unsqueeze")(0);
              }
          }
          
          py::list batched_info; batched_info.append(info);
          py::object out = py::cast(&self).attr("run_vectorized")(batched_obs, batched_info, agent_network, trajectory_actions, exploration);
          
          return py::make_tuple(
              out.attr("root_values")[py::cast(0)],
              out.attr("exploratory_policy")[py::cast(0)],
              out.attr("target_policy")[py::cast(0)],
              out.attr("best_actions")[py::cast(0)],
              py::dict() // metadata
          );
      }, py::arg("obs"), py::arg("info"), py::arg("agent_network"), py::arg("trajectory_actions") = py::none(), py::arg("exploration") = true)
      .def("run_vectorized", [](ModularSearch &self, py::object batched_obs, py::object batched_info, py::object agent_network, py::object trajectory_actions, bool exploration) {
          int B = py::len(batched_obs);
          
          // 1. Initial Inference
          py::object outputs = agent_network.attr("obs_inference")(batched_obs);
          // Explicit cpu/detach/double/numpy to avoid reading float32 bytes as float64
          // (py::array_t<double>::unchecked() is dtype-blind and causes UB on float32 buffers).
          py::array_t<double> values = outputs.attr("value")
              .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
              .cast<py::array_t<double>>();
          auto values_view = values.unchecked<1>();

          py::object policy_obj = outputs.attr("policy");
          py::array_t<double> policy_probs;
          if (py::hasattr(policy_obj, "probs") && !py::getattr(policy_obj, "probs").is_none()) {
              policy_probs = py::getattr(policy_obj, "probs")
                  .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
                  .cast<py::array_t<double>>();
          } else {
              policy_probs = py::module_::import("torch").attr("softmax")(
                      py::getattr(policy_obj, "logits"), -1)
                  .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
                  .cast<py::array_t<double>>();
          }
          auto probs_view = policy_probs.unchecked<2>();
          int num_actions = static_cast<int>(probs_view.shape(1));
          // to_play is now extracted per-environment from the info dictionary

          py::object network_state = outputs.attr("network_state");
          py::list unbatched_states = network_state.attr("unbatch")();
          
          py::list info_list;
          if (py::isinstance<py::dict>(batched_info)) {
              // Convert dict-of-tensors to list-of-dicts
              py::object unbatch_func = py::module_::import("utils.utils").attr("unbatch_dict");
              info_list = unbatch_func(batched_info);
          } else {
              info_list = batched_info.cast<py::list>();
          }

          // 2. Setup B engines
          std::vector<std::unique_ptr<ModularSearch>> engines;
          engines.reserve(B);
          std::vector<std::unordered_map<int64_t, py::object>> state_registries(B);

          for (int b = 0; b < B; ++b) {
              engines.push_back(std::make_unique<ModularSearch>(
                  self.get_search_config(), self.get_scoring_config(),
                  self.get_root_selection_config(), self.get_decision_selection_config(),
                  self.get_chance_selection_config(), self.get_backprop_config(),
                  self.get_root_scoring_type(), self.get_decision_scoring_type(),
                  self.get_root_selection_type(), self.get_decision_selection_type(),
                  self.get_chance_selection_type(), self.get_backprop_method()));
              
              std::vector<double> priors(num_actions);
              for (int a = 0; a < num_actions; ++a) priors[a] = probs_view(b, a);
              int to_play;
              py::object player_raw = info_list[b].attr("get")("player");
              if (py::hasattr(player_raw, "item")) {
                  to_play = player_raw.attr("item")().cast<int>();
              } else {
                  to_play = player_raw.cast<int>();
              }

              py::object legal = info_list[b].attr("get")("legal_moves", py::none());
              std::vector<int> allowed;
              if (!legal.is_none()) {
                  if (py::hasattr(legal, "tolist")) {
                      allowed = legal.attr("tolist")().cast<std::vector<int>>();
                  } else {
                      allowed = legal.cast<std::vector<int>>();
                  }
              }

              // Handle 0 is initial unbatched state
              state_registries[b][0] = unbatched_states[b];
              engines[b]->initialize_root(priors, to_play, 0, values_view(b), allowed);
          }

          // 3. Iterative interleaved search
          int num_sims = self.get_search_config().num_simulations;
          int sims_done = 0;
          int search_batch_size = self.get_search_config().default_batch_size;

          while (sims_done < num_sims) {
              std::vector<LeafBatchRequest> requests(B);
              std::vector<int> hidden_req_counts(B, 0);
              std::vector<int> afterstate_req_counts(B, 0);
              int total_hidden = 0;
              int total_afterstate = 0;

              for (int b = 0; b < B; ++b) {
                  requests[b] = engines[b]->step_search_until_leaves(search_batch_size);
                  hidden_req_counts[b] = static_cast<int>(requests[b].hidden_request_ids.size());
                  afterstate_req_counts[b] = static_cast<int>(requests[b].afterstate_request_ids.size());
                  total_hidden += hidden_req_counts[b];
                  total_afterstate += afterstate_req_counts[b];
              }

              if (total_hidden == 0 && total_afterstate == 0) break;

              // Batched Hidden Inference
              if (total_hidden > 0) {
                  py::list hidden_states;
                  py::list hidden_actions;
                  for (int b = 0; b < B; ++b) {
                      for (std::size_t i = 0; i < requests[b].hidden_request_ids.size(); ++i) {
                          int64_t handle = requests[b].hidden_parent_state_handles[i];
                          hidden_states.append(state_registries[b][handle]);
                          hidden_actions.append(requests[b].hidden_actions[i]);
                      }
                  }

                  py::object first_state = hidden_states[0];
                  py::object batched_hidden_state;
                  if (py::hasattr(py::type::of(first_state), "batch")) {
                      batched_hidden_state = py::type::of(first_state).attr("batch")(hidden_states);
                  } else {
                      batched_hidden_state = py::module_::import("torch").attr("stack")(hidden_states);
                  }

                  py::object h_outputs = agent_network.attr("hidden_state_inference")(
                      batched_hidden_state,
                      py::module_::import("torch").attr("tensor")(hidden_actions, py::arg("device") = py::getattr(agent_network, "device", py::none()))
                  );

                  py::array_t<double> h_values = h_outputs.attr("value")
                      .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
                      .cast<py::array_t<double>>();
                  py::array_t<double> h_rewards = h_outputs.attr("reward")
                      .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
                      .cast<py::array_t<double>>();

                  // Use probs instead of logits for MCTS priors
                  py::object h_policy_obj = h_outputs.attr("policy");
                  py::array_t<double> h_probs;
                  if (py::hasattr(h_policy_obj, "probs") && !py::getattr(h_policy_obj, "probs").is_none()) {
                      h_probs = py::getattr(h_policy_obj, "probs")
                          .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
                          .cast<py::array_t<double>>();
                  } else {
                      h_probs = py::module_::import("torch").attr("softmax")(
                              py::getattr(h_policy_obj, "logits"), -1)
                          .attr("cpu")().attr("detach")().attr("double")().attr("numpy")()
                          .cast<py::array_t<double>>();
                  }

                  py::array_t<int> h_to_plays = h_outputs.attr("to_play")
                      .attr("cpu")().attr("detach")().attr("int")().attr("numpy")()
                      .cast<py::array_t<int>>();
                  py::list h_unbatched_states = h_outputs.attr("network_state").attr("unbatch")();

                  auto h_values_view = h_values.unchecked<1>();
                  auto h_rewards_view = h_rewards.unchecked<1>();
                  auto h_probs_view = h_probs.unchecked<2>();
                  auto h_to_plays_view = h_to_plays.unchecked<1>();

                  int global_idx = 0;
                  for (int b = 0; b < B; ++b) {
                      search::HiddenInferenceUpdateBatch update;
                      update.num_actions = num_actions;
                      for (int i = 0; i < hidden_req_counts[b]; ++i) {
                          int64_t next_handle = static_cast<int64_t>(state_registries[b].size());
                          state_registries[b][next_handle] = h_unbatched_states[global_idx];

                          update.request_ids.push_back(requests[b].hidden_request_ids[i]);
                          update.next_state_handles.push_back(next_handle);
                          update.rewards.push_back(h_rewards_view(global_idx));
                          update.values.push_back(h_values_view(global_idx));
                          update.to_plays.push_back(h_to_plays_view(global_idx));
                          
                          for (int a = 0; a < num_actions; ++a) update.priors.push_back(h_probs_view(global_idx, a));
                          global_idx++;
                      }
                      engines[b]->update_leaves_and_backprop(update, {});
                  }
              }

              // TODO: Afterstate inference if total_afterstate > 0
              
              sims_done += search_batch_size;
          }

          // 4. Extraction
          py::array_t<float> out_target = py::array_t<float>({B, num_actions});
          py::array_t<float> out_exploratory = py::array_t<float>({B, num_actions});
          py::array_t<int64_t> out_best = py::array_t<int64_t>({B});
          py::array_t<float> out_values = py::array_t<float>({B});
          
          auto target_view = out_target.mutable_unchecked<2>();
          auto exploratory_view = out_exploratory.mutable_unchecked<2>();
          auto best_view = out_best.mutable_unchecked<1>();
          auto val_view = out_values.mutable_unchecked<1>();

          for (int b = 0; b < B; ++b) {
              std::vector<double> visits = engines[b]->root_child_visits();
              double sum_v = 0;
              for (double v : visits) sum_v += v;
              if (sum_v < 1e-8) sum_v = 1.0;

              int best_a = 0;
              double max_v = -1.0;
              const int n_visits = static_cast<int>(visits.size());
              for (int a = 0; a < num_actions; ++a) {
                  // Guard against visits.size() < num_actions (e.g. when allowed_actions
                  // is a strict subset and Node::expand sizes child_visits to |allowed|).
                  const double v = (a < n_visits) ? visits[a] : 0.0;
                  const float p = static_cast<float>(v / sum_v);
                  target_view(b, a) = p;
                  exploratory_view(b, a) = p;
                  if (v > max_v) {
                      max_v = v;
                      best_a = a;
                  }
              }
              best_view(b) = best_a;
              val_view(b) = static_cast<float>(engines[b]->root_value());
          }

          py::object torch_module = py::module_::import("torch");
          py::object SearchOutput = py::module_::import("search.aos_search.search_output").attr("SearchOutput");
          return SearchOutput(
              torch_module.attr("as_tensor")(out_target),
              torch_module.attr("as_tensor")(out_exploratory),
              torch_module.attr("as_tensor")(out_best),
              torch_module.attr("as_tensor")(out_values)
          );

      }, py::arg("batched_obs"), py::arg("batched_info"), py::arg("agent_network"), py::arg("trajectory_actions") = py::none(), py::arg("exploration") = true)
      .def("get_search_config", &ModularSearch::get_search_config)
      .def("get_scoring_config", &ModularSearch::get_scoring_config)
      .def("get_root_selection_config", &ModularSearch::get_root_selection_config)
      .def("get_decision_selection_config", &ModularSearch::get_decision_selection_config)
      .def("get_chance_selection_config", &ModularSearch::get_chance_selection_config)
      .def("get_backprop_config", &ModularSearch::get_backprop_config)
      .def("get_root_scoring_type", &ModularSearch::get_root_scoring_type)
      .def("get_decision_scoring_type", &ModularSearch::get_decision_scoring_type)
      .def("get_root_selection_type", &ModularSearch::get_root_selection_type)
      .def("get_decision_selection_type", &ModularSearch::get_decision_selection_type)
      .def("get_chance_selection_type", &ModularSearch::get_chance_selection_type)
      .def("get_backprop_method", &ModularSearch::get_backprop_method)
      .def("root_value", &ModularSearch::root_value)
      .def("root_child_priors", &ModularSearch::root_child_priors)
      .def("root_child_values", &ModularSearch::root_child_values)
      .def("root_child_visits", &ModularSearch::root_child_visits)
      .def("select_root_action", &ModularSearch::select_root_action,
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
  m.def(
      "compute_scores",
      static_cast<std::vector<double> (*)(
          ScoringMethodType,
          const NodeArena &,
          int,
          const MinMaxStats &,
          const ScoringConfig &)>(&search::compute_scores),
      py::arg("type"), py::arg("arena"), py::arg("node_index"),
      py::arg("min_max_stats"), py::arg("config"));
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
