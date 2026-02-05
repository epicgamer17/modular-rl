---
description: Runs unit tests and performance benchmarks to ensure no regressions in speed or accuracy.
---

# Goal
Verify that recent changes have not broken existing functionality or degraded Steps/Second performance.

# Steps
1. **Unit Testing**
   - Run the project's test suite using the launcher to ensure correct environment variables:
     ```bash
     python launcher.py pytest tests/
     ```
   - If any test fails, STOP and report the error log.

2. **Performance Benchmarking**
   - Run the comparison script:
     ```bash
     python launcher.py python -m tests.benchmarks.benchmark_comparison
     ```

3. **Analysis**
   - Compare the output against the known baseline (from `tests/benchmarks/baseline_results.json` if it exists, or previous logs).
   - Check for:
     - **Speed Regression:** Is `steps/s` > 10% lower than before?
     - **Accuracy:** Did the random seed evaluation score drop?

4. **Report**
   - Summarize findings: "Tests passed. Performance is stable (120 steps/s)." or "WARNING: Performance dropped by 15%."