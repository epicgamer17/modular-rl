from core import PipelineComponent, Blackboard


class PPODecoratorComponent(PipelineComponent):
    """
    Sequential decorator that adds log_prob to the action_metadata.
    MUST be placed after a selector that provides 'policy_dist' in metadata.
    """

    def execute(self, blackboard: Blackboard) -> None:
        action = blackboard.meta["action_tensor"]
        metadata = blackboard.meta["action_metadata"]

        dist = metadata.get("policy_dist")
        if dist is not None:
            # We assume action and dist are compatible in shape (handled by selector)
            metadata["log_prob"] = dist.log_prob(action).detach().cpu()

        # 'value' is already added by the selectors from result.value
