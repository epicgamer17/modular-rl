import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python lint.py <file_or_directory>")
        sys.exit(1)

    target = sys.argv[1]

    # Resolve workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent.resolve()
    target_path = workspace_root / target

    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist.")
        sys.exit(1)

    print(f"Linting {target}...")

    # Run pylint
    # We use 'pylint' assuming it is installed in the environment
    cmd = ["pylint", str(target_path)]

    try:
        result = subprocess.run(cmd, cwd=workspace_root, capture_output=True, text=True)
        print(result.stdout)
        # Pylint returns non-zero exit codes for issues, but we might just want to see the output
        if result.returncode != 0:
            print(f"Pylint finished with score/status: {result.returncode}")
    except FileNotFoundError:
        print("Error: 'pylint' is not installed or not in PATH.")


if __name__ == "__main__":
    main()
