---
name: code-quality
description: Runs static analysis (pylint) on Python files to ensure code quality standards are met. Use this before finalizing any code changes or when the user asks to "check the code".
---

# Goal
Verify Python code against project standards using `pylint`.

# Instructions
1.  Identify the target file or directory.
2.  Run the linting script.
3.  Report any violations (e.g., missing docstrings, unused imports).

# Constraints
* Focus on `pylint` errors as the primary source of truth (as referenced by `pylint_output.txt` in the repo).

# Examples
**User:** "I updated the PPO agent. Can you check for errors?"
**Agent:** (Calls skill with `target="agents/ppo.py"`)