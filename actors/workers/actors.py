import time
import torch
from typing import Any, Callable, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from data.samplers.sequence import Sequence
from actors.action_selectors.selectors import BaseActionSelector
from actors.action_selectors.types import InferenceResult
from actors.action_selectors.policy_sources import (
    BasePolicySource,
    NetworkPolicySource,
    SearchPolicySource,
)
from data.storage.circular import ModularReplayBuffer
from modules.agent_nets.modular import ModularAgentNetwork
from envs.factories.wrappers import wrap_recording
import numpy as np


# This file is deprecated. Legacy actors have been replaced by component-based actors 
# using the BlackboardEngine and specifiche pipeline components.
# See components/environment.py and components/actor_logic.py for replacements.
