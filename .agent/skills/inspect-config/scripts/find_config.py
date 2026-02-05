import sys
import os
import glob
from pathlib import Path

# CONSTANTS
AGENT_CONFIG_DIR = "agent_configs"
GAME_CONFIG_DIR = "game_configs"


def main():
    if len(sys.argv) < 3:
        print("Usage: python find_config.py <agent|game> <query>")
        sys.exit(1)

    config_type = sys.argv[1].lower()
    query = sys.argv[2].lower()

    # Determine target directory
    # This script is in .agent/skills/inspect-config/scripts/
    workspace_root = Path(__file__).parent.parent.parent.parent.resolve()

    if config_type == "agent":
        target_dir = workspace_root / AGENT_CONFIG_DIR
    elif config_type == "game":
        target_dir = workspace_root / GAME_CONFIG_DIR
    else:
        print("Error: config_type must be 'agent' or 'game'")
        sys.exit(1)

    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist.")
        sys.exit(1)

    # Simple fuzzy search for files
    matches = []
    for file_path in target_dir.glob("*.py"):
        if query in file_path.name.lower():
            matches.append(file_path)

    if not matches:
        print(f"No config found for '{query}' in {config_type} configs.")
        print(f"Available files: {[f.name for f in target_dir.glob('*.py')]}")
        sys.exit(0)

    # Return the first match's content
    best_match = matches[0]
    print(f"Found config: {best_match.name}\n")
    print("-" * 40)
    try:
        print(best_match.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
