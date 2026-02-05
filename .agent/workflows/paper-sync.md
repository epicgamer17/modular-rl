---
description: Extracts latest experimental results and formats them as LaTeX table rows.
---

# Goal
Parse recent experiment logs and generate LaTeX code for the research paper.

# Steps
1. **Locate Data**
   - Find the most recent experiment log in `experiments/` (e.g., `experiments/rainbow_hyperopt/...`).
   - Read the final CSV or Pickle results.

2. **Format LaTeX**
   - Extract `Mean Reward` and `Standard Deviation`.
   - Format into a LaTeX row:
     ```latex
     AlgorithmName & $\mu \pm \sigma$ & \textbf{Best Score} \\
     ```

3. **Update File (Optional)**
   - If the user consents, append this directly to a `results.tex` file or `experiments/arxiv-style-master/paper.tex`.
   - Otherwise, print the LaTeX snippet in the chat for copy-pasting.