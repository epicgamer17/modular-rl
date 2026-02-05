---
name: verify-compliance
description: Compares current code changes (git diff) against the project's engineering rules located in .agent/rules/. Use this to perform a self-check before finalizing tasks.
---

# Goal
Ensure recent code changes adhere to the project's specific guidelines (e.g., PyTorch practices, Error Handling, RL specific naming).

# Instructions
1.  Call the `get_diff_and_rules.py` script to retrieve the current code changes and the text of the project rules.
2.  **Analyze the output:**
    * Does the added code use `var` instead of specific types? (Violates `type-hinting.md`)
    * Are tensor shapes asserted? (Checks `pytorch-practices.md`)
    * Is there proper logging? (Checks `error-handling.md`)
3.  If violations are found, list them and suggest fixes. If clear, report "Compliance Check Passed".

# Examples
**User:** "Check if my changes to the MuZero agent follow the rules."
**Agent:** (Calls skill) -> (Reads output) -> "I noticed you used a bare `try-except` block in `muzero.py`. According to `error-handling.md`, we must catch specific exceptions. I will fix this."