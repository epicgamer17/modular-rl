import torch
import torch.nn as nn
import copy
import pickle


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def obs_inference(self, x):
        return self.linear(x)

    def compile(self):
        self.obs_inference = torch.compile(self.obs_inference)


model = MyModel()
model.compile()

# This should fail
try:
    pickle.dumps(model)
    print("Pickle successful (unexpected)")
except Exception as e:
    print(f"Pickle failed as expected: {e}")


# Now our fix
def get_uncompiled_model(m):
    import copy

    m_copy = copy.copy(m)
    m_copy.__dict__ = copy.copy(
        m.__dict__
    )  # Ensure __dict__ is copied so we don't mutate original
    for attr in ["obs_inference"]:
        if attr in m_copy.__dict__:
            del m_copy.__dict__[attr]
    return m_copy


uncompiled = get_uncompiled_model(model)
try:
    pickle.dumps(uncompiled)
    print(
        "Pickle uncompiled successful! Parameters shared?:",
        uncompiled.linear.weight is model.linear.weight,
    )
except Exception as e:
    print(f"Pickle uncompiled failed: {e}")
