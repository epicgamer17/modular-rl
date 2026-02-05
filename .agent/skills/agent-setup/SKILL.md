---
name: agent-setup
description: Instructions for the agent on how to run python scripts correctly in this environment.
---

# Agent Environment Setup

When running Python scripts that involve PyTorch, Ray, or Compilation in this workspace, you **MUST** set the `ANTIGRAVITY_AGENT` environment variable to `1`.

This ensures that:
1.  Isolated temporary directories are created for `torch.compile` artifacts.
2.  Permission errors related to system temporary directories are avoided.
3.  Your runs do not conflict with or lock the User's persistent cache files.

## Usage

When using `run_command` or similar tools, prepend the variable:

```bash
ANTIGRAVITY_AGENT=1 python my_script.py
```

Do NOT set this permanently in a `.rc` file; just apply it to your runtime commands.
