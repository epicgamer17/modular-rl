import sys
import subprocess
import os
import glob
from pathlib import Path


def main():
    # Resolve workspace root
    # Script location: .agent/skills/verify-compliance/scripts/
    workspace_root = Path(__file__).parent.parent.parent.parent.resolve()
    rules_dir = workspace_root / ".agent" / "rules"

    print("=== 1. GETTING CODE CHANGES (GIT DIFF) ===\n")
    try:
        # Get unstaged changes
        diff_proc = subprocess.run(
            ["git", "diff"], cwd=workspace_root, capture_output=True, text=True
        )
        # Get staged changes
        diff_cached_proc = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
        )

        full_diff = diff_proc.stdout + diff_cached_proc.stdout

        if not full_diff.strip():
            print("[No changes detected in git. Verify you have modified files.]")
        else:
            # Limit diff size to prevent context overflow if necessary
            print(full_diff[:10000])
            if len(full_diff) > 10000:
                print("\n...[Diff truncated]...")
    except Exception as e:
        print(f"Error getting git diff: {e}")

    print("\n\n=== 2. READING PROJECT RULES ===\n")
    if not rules_dir.exists():
        print(f"Error: Rules directory not found at {rules_dir}")
        sys.exit(1)

    # Read all markdown files in the rules folder
    for rule_file in rules_dir.glob("*.md"):
        print(f"--- Rule File: {rule_file.name} ---")
        try:
            content = rule_file.read_text(encoding="utf-8")
            print(content)
            print("\n")
        except Exception as e:
            print(f"Could not read {rule_file.name}: {e}")


if __name__ == "__main__":
    main()
