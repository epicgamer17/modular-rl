import torch
from typing import Optional, Dict, Any
from runtime.runtime import LearnerRuntime
from runtime.context import ExecutionContext
from .config import PPOConfig


class OnPolicyLearner(LearnerRuntime):
    """
    Learner runtime for on-policy algorithms like PPO.
    Handles GAE computation and multiple epochs of updates.
    """

    def __init__(
        self,
        train_graph,
        config: PPOConfig,
        ac_net,
        actor_runtime,
        buffer_id: str = "main",
    ):
        super().__init__(train_graph)
        self.config = config
        self.ac_net = ac_net
        self.actor_runtime = actor_runtime
        self.buffer_id = buffer_id

    def update_step(self, context: Optional[ExecutionContext] = None):
        """
        Execute the PPO update loop:
        1. Compute GAE advantages for the whole buffer.
        2. Iterate over minibatches for several epochs.
        3. Clear the buffer.
        """
        import time

        start_time = time.perf_counter()

        buffer = context.get_buffer(self.buffer_id)
        if len(buffer) < self.config.rollout_steps:
            return {}  # Not enough data yet

        device = next(self.ac_net.parameters()).device
        last_obs = torch.as_tensor(
            self.actor_runtime.last_obs, dtype=torch.float32, device=device
        )
        last_done = torch.as_tensor(
            self.actor_runtime.last_done, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            _, next_value = self.ac_net(last_obs.unsqueeze(0))
            next_value = next_value.squeeze(0)

        # 1. LR Annealing
        if self.config.anneal_lr:
            frac = 1.0 - (context.actor_step / self.config.total_steps)
            frac = max(0.0, frac)
            new_lr = frac * self.config.learning_rate
            opt_state = context.get_optimizer(self.config.optimizer_handle)
            for param_group in opt_state.optimizer.param_groups:
                param_group["lr"] = new_lr

        # 2. Compute GAE
        buffer.compute_returns_advantages(
            next_value=next_value,
            next_done=last_done,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # 3. PPO Update Loop (Epochs & Minibatches)
        all_results = []
        for epoch in range(self.config.epochs):
            epoch_kl = []
            for minibatch in buffer.iterate_minibatches(self.config.minibatch_size):
                res = super().update_step(batch=minibatch, context=context)
                all_results.append(res)

                # 3.1 Extract KL for early stopping
                # The 'opt' node returns the metrics from 'ppo' objective
                opt_res = res.get("opt", {})
                if isinstance(opt_res, dict) and "approx_kl" in opt_res:
                    epoch_kl.append(opt_res["approx_kl"])

            # 3.2 Check for Early Stopping
            if self.config.target_kl is not None and epoch_kl:
                mean_kl = sum(epoch_kl) / len(epoch_kl)
                if mean_kl > self.config.target_kl:
                    # Optional: log or record early stopping
                    break

        # 4. Clear the buffer
        buffer.clear()

        if not all_results:
            return {}

        # 5. Aggregate Metrics
        # Flatten all_results because some nodes (like metrics sink) return dictionaries
        flattened_results = []
        for res in all_results:
            flat = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    flat.update(v)
                else:
                    flat[k] = v
            flattened_results.append(flat)

        if not flattened_results:
            return {}

        agg_metrics = {}
        # Aggregate mean across all minibatches/epochs for this update
        for k in flattened_results[0].keys():
            values = [
                r[k] for r in flattened_results if isinstance(r.get(k), (int, float))
            ]
            if values:
                agg_metrics[k] = sum(values) / len(values)

        # 6. Add Diagnostics
        end_time = time.perf_counter()
        total_steps = self.config.rollout_steps * self.config.num_envs
        agg_metrics["sps"] = total_steps / (end_time - start_time + 1e-8)
        agg_metrics["episodic_return"] = self.actor_runtime.last_episode_return
        agg_metrics["episodic_length"] = self.actor_runtime.last_episode_length

        return agg_metrics
