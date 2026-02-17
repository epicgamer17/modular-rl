import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from configs.agents.muzero import MuZeroConfig

    print("Successfully imported MuZeroConfig")
except Exception as e:
    print(f"Failed to import MuZeroConfig: {e}")
    import traceback

    traceback.print_exc()
