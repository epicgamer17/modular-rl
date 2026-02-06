---
description: structured commit of changes.
---
// turbo-all
1. Analyze Changes
   - Run `git status` and `git diff` to understand changes.
   - Summarize findings for the commit message.

2. Compliance Check
   - Ensure no debug code (e.g., `print`, `pdb`, `set_trace`) is present in the current diff.
   - If found, remove them or ask the user for confirmation.

3. Generate Message
   - Create a commit message following Conventional Commits format (e.g., `feat(muzero): ...`).
   - Present the message to the user for approval.

4. Execute Local Commit
   - Proactively stage and commit all changes:
     `git add . && git commit -m "<approved_message>"`
   - IMPORTANT: Do not ask for clarification on complex repository states (e.g., rebases, detached HEAD, special branches) unless the git command explicitly fails. Assume a standard commit on the current state is the intended action.

5. Handover
   - Inform the user: "Commit created locally. Please review and push from GitHub Desktop."
