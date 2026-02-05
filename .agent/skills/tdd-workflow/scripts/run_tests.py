import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <test_path_or_keyword>")
        sys.exit(1)

    target = sys.argv[1]

    # Resolve workspace root (assuming standard .agent structure)
    workspace_root = Path(__file__).parent.parent.parent.parent.resolve()

    print(f"=== TDD Step: Running Tests for '{target}' ===")

    # Construct pytest command with verbose output to see specific failures
    cmd = ["pytest", str(target), "-v", "-s"]

    try:
        result = subprocess.run(cmd, cwd=workspace_root, text=True, capture_output=True)

        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            print("\n✅ STATUS: GREEN (Tests Passed)")
        else:
            print("\n❌ STATUS: RED (Tests Failed)")

        sys.exit(result.returncode)

    except Exception as e:
        print(f"Error executing pytest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
