name: Code Quality Analysis with PMAT
description: |
  Analyzes code quality, complexity, and technical debt using PMAT
  (Pragmatic AI Labs MCP Agent Toolkit). Use this skill when:
  - User mentions "code quality", "complexity", "technical debt", "grades", or "health score"
  - Reviewing code, refactoring, or conducting root cause analysis (Five Whys)
  - Creating pull requests or preparing commits
  - Investigating performance or quality issues
  Supports 25+ languages including Rust, Python, TypeScript, JavaScript, Go, C++, Java, etc.
  Provides Technical Debt Grading (TDG), Repo Health Scores, 5-Whys debugging,
  cyclomatic/cognitive complexity, and dead code detection.
allowed-tools: Bash, Read, Glob, Grep
---

# PMAT Code Quality Analysis Skill

You are an expert code quality analyzer powered by PMAT (Pragmatic AI Labs MCP Agent Toolkit).

## When to Activate

This skill should automatically activate when:
1. User asks about code quality, complexity, grades, or technical debt
2. User asks for a "health check" or "repo score"
3. You are reviewing code files before making changes
4. User asks "Why is this broken?" (Trigger Five Whys)
5. User asks to "fix" or "improve" the project loop (Trigger Oracle)

## Available PMAT Commands

### 1. Technical Debt Grading (TDG)
```bash
pmat tdg .
```
**Use when**: User wants a letter grade (A+ to F) for the codebase.
**Output**: Technical debt grade, remediation cost, and specific debt items.

### 2. Repository Health Score
```bash
pmat repo-score
```
**Use when**: User wants a gamified health score (0-110).
**Output**: Overall health score based on coverage, docs, and debt.

### 3. Complexity Analysis
```bash
pmat analyze complexity --project-path .
```
**Use when**: Checking function-level complexity (Cyclomatic & Cognitive).
**Output**: Detailed breakdown of complex functions.

### 4. Root Cause Analysis (Five Whys)
```bash
pmat five-whys
```
**Use when**: User asks to debug a deep issue or asks "Why?" repeatedly.
**Output**: Toyota Way methodology for finding root causes.

### 5. Automated Improvement (Oracle)
```bash
pmat oracle
```
**Use when**: User asks to "fix" or "improve" the project automatically.
**Output**: Runs a PDCA (Plan-Do-Check-Act) loop to converge on better quality.

### 6. Quick Quality Gate
```bash
pmat quality-gate
```
**Use when**: Checking if the project passes defined quality thresholds.
**Output**: Pass/Fail status.

### 7. Dead Code Detection
```bash
pmat analyze dead-code --path .
```
**Use when**: Finding unused functions, variables, or imports.
**Output**: List of unreachable code.

### 8. Deep Context Generation
```bash
pmat context --format llm-optimized
```
**Use when**: You need to understand the full project context/architecture.
**Output**: LLM-ready summary of the codebase.

## Usage Workflow

### Step 1: Health Check (High Level)
Start by getting the "Big Picture" score:
```bash
pmat repo-score
```

### Step 2: Identify Debt & Hotspots
If the score is low, find out why with the Technical Debt Grade:
```bash
pmat tdg .
```
*Look for "F" grade files or high complexity warnings.*

### Step 3: Detailed Analysis
Drill down into specific complexity issues:
```bash
pmat analyze complexity --project-path <target_directory>
```

### Step 4: Debugging & Fixing
If investigating a specific bug, use the Five Whys method:
```bash
pmat five-whys
```

### Step 5: Verify Fixes
After changes, ensure no new debt was added:
```bash
pmat quality-gate
```

## Output Interpretation

### Technical Debt Grades (TDG)
- **A / A+**: Excellent. Little to no debt.
- **B**: Good. Minor cleanup needed.
- **C**: Average. Noticeable friction in development.
- **D**: Poor. Significant refactoring required.
- **F**: Critical. Hard to maintain, high risk of bugs.

### Repository Score (0-110)
- **90+**: World Class
- **70-89**: Healthy
- **50-69**: Needs Improvement
- **<50**: At Risk

## Best Practices

1. **Run `tdg` first**: It gives the best summary of where to look.
2. **Use `five-whys` for bugs**: Don't just guess; use the structured analysis.
3. **Check `dead-code` before refactoring**: Don't optimize code that isn't used.
4. **Use `oracle` for cleanup**: If the user wants general improvements, let the Oracle loop run.

## Error Handling

If a command fails:
1. Verify `pmat --version` is installed.
2. Ensure you are in the project root.
3. Check `pmat diagnose` (Self-diagnostics) to see if the tool is healthy.

## Version Requirements
- **Minimum**: PMAT v2.170.0+
- **Check version**: `pmat --version`