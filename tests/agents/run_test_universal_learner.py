from test_universal_learner import *
import torch

if __name__ == "__main__":
    # Mocking necessary parts for manual run
    test_universal_learner_init_sets_fields()
    test_universal_learner_step_calls_optimizer_and_callbacks()
    test_universal_learner_early_stop_iteration_breaks_loop()
    print("test_universal_learner.py manual run passed!")
