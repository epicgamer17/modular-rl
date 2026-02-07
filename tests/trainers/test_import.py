import sys
from unittest.mock import MagicMock

sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

print("Mocks applied, importing dependencies...", flush=True)

print("Importing executors...", flush=True)
import executors

print("Importing muzero_learner...", flush=True)
import agents.muzero_learner

print("Importing muzero_policy...", flush=True)
import agents.muzero_policy

print("Importing actors...", flush=True)
import agents.actors

print("Importing Network...", flush=True)
from modules.agent_nets.muzero import Network

print("Importing StatTracker...", flush=True)
from stats.stats import StatTracker

print("Importing muzero_trainer...", flush=True)
import trainers.muzero_trainer as mt

print("Import successful!", flush=True)
