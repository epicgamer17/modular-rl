import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_coverage.py <module_to_check>")
        sys.exit(1)

    target_module = sys.argv[1]

    # Resolve workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent.resolve()

    print(f"=== TDD Step: Checking Coverage for '{target_module}' ===")

    # Run pytest with coverage
    # Example: pytest --cov=agents/muzero tests/muzero/
    cmd = [
        "pytest",
        f"--cov={target_module}",
        "--cov-report=term-missing",
        "tests/",  # Runs all tests to ensure integration coverage is caught
    ]

    try:
        result = subprocess.run(cmd, cwd=workspace_root, text=True, capture_output=True)
        print(result.stdout)

        # Simple parsing logic could be added here to enforce the 80% rule programmatically
        if "FAIL" in result.stdout:
            print("\n⚠️  Warning: Tests failed during coverage check.")

    except Exception as e:
        print(f"Error checking coverage: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
