
import torch
from components.valves import GradientScaleValve
from core import Blackboard

def reproduce():
    blackboard = Blackboard()
    valve = GradientScaleValve(key="missing_tensor", scale=0.5)
    try:
        valve.execute(blackboard)
    except KeyError as e:
        print(f"Caught KeyError: {repr(e)}")
        print(f"Message: {str(e)}")
        print(f"Args: {e.args}")
    except Exception as e:
        print(f"Caught other error: {type(e)} - {e}")

if __name__ == "__main__":
    reproduce()
