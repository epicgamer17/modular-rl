from search.search_selectors import SamplingSelection, TopScoreSelection
from search.backpropogation import (
    AverageDiscountedReturnBackpropagator,
    MinimaxBackpropagator,
)
from search.initial_searchsets import SelectAll, SelectTopK
from search.modular_search import SearchAlgorithm
from search.pruners import SequentialHalvingPruning, NoPruning, AlphaBetaPruning
from search.prior_injectors import (
    ActionTargetInjector,
    DirichletInjector,
    GumbelInjector,
)
from search.root_policies import (
    CompletedQValuesRootPolicy,
    VisitFrequencyPolicy,
    BestActionRootPolicy,
)
from search.scoring_methods import (
    GumbelScoring,
    LeastVisitedScoring,
    PriorScoring,
    UCBScoring,
    QValueScoring,
)


def create_mcts(config, device, num_actions) -> SearchAlgorithm:
    if config.gumbel:
        return SearchAlgorithm(
            config,
            device,
            num_actions,
            root_selection_strategy=TopScoreSelection(LeastVisitedScoring()),
            decision_selection_strategy=TopScoreSelection(GumbelScoring(config)),
            chance_selection_strategy=SamplingSelection(PriorScoring()),
            root_target_policy=CompletedQValuesRootPolicy(config, device, num_actions),
            root_exploratory_policy=VisitFrequencyPolicy(config, device, num_actions),
            prior_injectors=[ActionTargetInjector(), GumbelInjector()],
            root_searchset=SelectTopK(),
            internal_searchset=SelectAll(),
            pruning_method=SequentialHalvingPruning(),
            internal_pruning_method=NoPruning(),
            backpropagator=AverageDiscountedReturnBackpropagator(),
        )
    else:
        return SearchAlgorithm(
            config,
            device,
            num_actions,
            root_selection_strategy=TopScoreSelection(UCBScoring()),
            decision_selection_strategy=TopScoreSelection(UCBScoring()),
            chance_selection_strategy=SamplingSelection(PriorScoring()),
            root_target_policy=VisitFrequencyPolicy(config, device, num_actions),
            root_exploratory_policy=VisitFrequencyPolicy(config, device, num_actions),
            prior_injectors=[ActionTargetInjector(), DirichletInjector()],
            root_searchset=SelectAll(),
            internal_searchset=SelectAll(),
            pruning_method=NoPruning(),
            internal_pruning_method=NoPruning(),
            backpropagator=AverageDiscountedReturnBackpropagator(),
        )
        # return SearchAlgorithm(
        #     config=config,
        #     device=device,
        #     num_actions=num_actions,
        #     # Selection: LeastVisited prioritizes visits=0 (unexpanded) then min visits (deepening)
        #     root_selection_strategy=TopScoreSelection(
        #         LeastVisitedScoring(), tiebreak_scoring_method=UCBScoring()
        #     ),
        #     decision_selection_strategy=TopScoreSelection(
        #         LeastVisitedScoring(), tiebreak_scoring_method=UCBScoring()
        #     ),
        #     chance_selection_strategy=SamplingSelection(PriorScoring()),
        #     # Output: Best Action based on Minimax Value
        #     root_target_policy=BestActionRootPolicy(config, device, num_actions),
        #     root_exploratory_policy=VisitFrequencyPolicy(config, device, num_actions),
        #     # Setup
        #     prior_injectors=[],
        #     root_searchset=SelectAll(),
        #     internal_searchset=SelectAll(),
        #     # Pruning: Alpha Beta on both Root and Internal
        #     pruning_method=AlphaBetaPruning(),
        #     internal_pruning_method=AlphaBetaPruning(),
        #     # Backprop: Minimax
        #     backpropagator=MinimaxBackpropagator(),
        # )
