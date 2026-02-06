import torch
from utils import action_mask, normalize_policies, current_timestamp, get_legal_moves
from utils.utils import clip_low_prob_actions, get_lr_scheduler
from agents.agent import BaseAgent
from modules.utils import get_optimal_device
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD

from modules.policy_imitation import SupervisedNetwork
from replay_buffers.nfsp_reservoir_buffer import NFSPReservoirBuffer


class PolicyImitationAgent(BaseAgent):
    def __init__(
        self,
        env,
        config,
        name=f"policy_imitation_{current_timestamp():.1f}",
        device: torch.device = get_optimal_device(),
        from_checkpoint=False,
    ):
        super().__init__(env, config, name, device, from_checkpoint=from_checkpoint)

        self.replay_buffer = NFSPReservoirBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.observation_dtype,
            max_size=self.config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=self.config.minibatch_size,
        )

        self.model = SupervisedNetwork(
            config,
            self.num_actions,
            torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
        ).to(self.device)

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)

    def select_actions(self, predictions):
        distribution = torch.distributions.Categorical(probs=predictions)
        # print("Probabilities", predictions)
        selected_action = distribution.sample()
        return selected_action

    def predict(
        self,
        state,
        info: dict = None,
    ):
        state_input = self.preprocess(state)
        policy = self.model(inputs=state_input)
        if "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            policy = action_mask(policy, legal_moves, mask_value=0, device=self.device)
            # print("Original", policy)
            policy = normalize_policies(policy)
            policy = clip_low_prob_actions(policy, self.config.clip_low_prob)
            policy = normalize_policies(policy)
            # print("Masked for low probs", policy)
        return policy

    def learn(self):
        for training_iteration in range(self.config.training_iterations):
            sample = self.replay_buffer.sample()
            observations = sample["observations"]
            # targets = torch.from_numpy(sample["targets"]).to(self.device)
            targets = sample["targets"].to(self.device)

            legal_moves_masks = sample["legal_moves_masks"].to(self.device)
            # recreate infos from action masks
            infos = [
                {"legal_moves": torch.nonzero(mask).view(-1).tolist()}
                for mask in legal_moves_masks
            ]

            policy = self.predict(observations, infos)
            loss = self.config.loss_function(policy, targets).mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clipnorm)
            self.optimizer.step()
            self.lr_scheduler.step()

            # RESET NOISE IF IM DOING THAT
        return loss.detach()
