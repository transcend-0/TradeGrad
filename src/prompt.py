TEXTLOSS_SYSTEM_PROMPT = '''You are an expert in optimizing trading strategies. Your goal is to {} the `combined_score` (0-100, above 50 is good).'''

REWARD_SYSTEM_PROMPT = TEXTLOSS_SYSTEM_PROMPT.format('maximize')

GRAD_EXPLOIT_USER_PROMPT_LARGE_REVISION = '''
# Task
Analyze the following strategies and their performance, and generate directions to aggressively EXPLOIT the successful patterns by taking a LARGE optimization step on the best strategy.

{reference_strategies_with_performance}

# Note
- Analyze what specific trading logic and implementation details were used in the strategies, how these details affected strategy performance, and identify the strongest patterns to amplify as well as the weakest components to replace.
- Propose BOLD, high-impact directions: restructuring the core trading logic, swapping out ineffective signals/filters/risk modules for proven alternatives from the references, or fusing multiple successful elements into a substantially upgraded variant. Do not be limited by the current structure when a more promising design is evident.
- You can get history data via `self.get_history_map` (multi-symbol) or `self.get_history` (single-symbol) — use whichever the current strategy code already uses. Available fields include `open`, `high`, `low`, `close`, `volume`.
- You can only modify the code within `### EDIT START` and `### EDIT END` markers. Do not change anything outside these markers.
- Do not give code, just provide directions on how to substantially improve the strategy.
'''.strip()

GRAD_EXPLOIT_USER_PROMPT_SMALL_REVISION = '''
# Task
Analyze the following strategies and their performance, and generate directions to conservatively EXPLOIT the successful patterns by taking a SMALL optimization step on the best strategy.

{reference_strategies_with_performance}

# Note
- Analyze what specific trading logic and implementation details were used in the strategies, how these details affected strategy performance, and identify the fine-grained tweaks that could nudge the current strategy in the direction of the better-performing references.
- Propose one or two LOCAL changes: small in extent, but not restricted in kind. A local change may add, remove or replace a condition, filter, indicator or sizing rule, as well as retune a parameter or lookback window. What makes it local is that it touches one part of the strategy rather than reworking the whole pipeline. Retuning an existing number is the weakest option available and is rarely the most useful one — when the evidence points at the logic, change the logic.
- You can get history data via `self.get_history_map` (multi-symbol) or `self.get_history` (single-symbol) — use whichever the current strategy code already uses. Available fields include `open`, `high`, `low`, `close`, `volume`.
- You can only modify the code within `### EDIT START` and `### EDIT END` markers. Do not change anything outside these markers.
- Do not give code, just provide directions on how to incrementally refine the strategy.
'''.strip()

SINGLE_STRATEGY_PERFORMANCE_TEMPLATE = '''
# Strategy {id}
## Code
```python
{strategy_code}
```

## Performance
{performance}'''

OPTIMIZER_SYSTEM_PROMPT = '''
You are an expert in optimizing trading strategies.
The user will provide you with a strategy code and its feedback.
Your task is to generate an improved version of the strategy code based on the feedback.
Generate the full code of the improved strategy within ```python ... ``` code block.
'''.strip()

OPTIMIZER_USER_PROMPT = '''
# Strategy Code to Optimize
```python
{strategy_code}
```

# Feedback
{feedback}

# Note
- You can only modify the code within `### EDIT START` and `### EDIT END` markers. Do not change anything outside these markers.
- Prioritize vectorized operations and efficient data handling when making improvements.
- Get history data via `self.get_history_map` (multi-symbol) or `self.get_history` (single-symbol) — use whichever the current strategy code already uses. Available fields include `open`, `high`, `low`, `close`, `volume`.
'''.strip()

OPTIMIZER_USER_PROMPT_SMALL_REVISION = '''
# Strategy Code to Optimize
```python
{strategy_code}
```

# Feedback
{feedback}

# Note
- You can only modify the code within `### EDIT START` and `### EDIT END` markers. Do not change anything outside these markers.
- Prioritize vectorized operations and efficient data handling when making improvements.
- Make one or two localized changes. They may be structural -- adding, removing or replacing a
  condition, filter, indicator or sizing rule -- as long as each stays confined to one part of the
  strategy. Changing only numeric values is allowed but is the weakest option; do not default to it
  when the feedback points at the logic itself.
- Output with SEARCH/REPLACE diff format.
Output Example:
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE
'''.strip()

OPTIMIZER_SYSTEM_PROMPT_SMALL_REVISION = '''
You are an expert in optimizing trading strategies.
The user will provide you with a strategy code and its feedback. You should make a LOCAL edit to the
strategy based on the feedback: small in extent, but not restricted in kind.

A local edit may add, remove or replace a condition, a filter, an indicator or a sizing rule, as well
as retune a number. What makes it local is that it touches one part of the strategy rather than
rewriting the whole thing -- it is not a rule that only the numbers may change. Retuning an existing
parameter is the weakest edit available and rarely the most useful one; prefer changing the logic
where the feedback points at the logic.

Output with SEARCH/REPLACE diff format.

Output Example:
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

Note:
- You can only modify the code within `### EDIT START` and `### EDIT END` markers. Do not change anything outside these markers.
- Do not repeat the markers `### EDIT START` and `### EDIT END` in the SEARCH/REPLACE blocks.
- Every block's SEARCH section must be copied **verbatim** from the original code, including indentation.
'''.strip()

MEMORY_INDIVIDUAL_SUMMARY_SYSTEM_PROMPT = '''
You are an expert in analyzing trading strategies.
'''.strip()

MEMORY_INDIVIDUAL_SUMMARY_USER_PROMPT = '''
# Strategy Code to Analyze
```python
{strategy_code}
```

## Performance
{performance}

# Task
Summarize the strengths and weaknesses of the strategy based on its code and performance. Keep the strategy summary concise but informative. Focus on:
1. What specific trading logic/signals and implementation details were used
2. How these details affected strategy performance (profitability, risk, etc.)
3. Implementation details that are relevant to the approach
4. Any evaluation feedback that provides insights

# Output Format
Strategy Name: [Short summary name of the strategy (up to 10 words)]
- Implementation: [Key trading logic and implementation details (1-2 sentences)]
- Performance: [Sharpe ratio, Max Drawdown, or other key metrics summary]
- Feedback: [Key insights from evaluation (1-2 sentences)
'''.strip()

MEMORY_GLOBAL_SUMMARY_SYSTEM_PROMPT = '''
You are an expert in analyzing trading strategy implementations and performance.
`combined_score` is ranged from 0 to 100, with higher values indicating better performance. And above 50 is considered good.
Extract evidence-based success and failure patterns, and consolidate them into reusable "Past Experience".
'''.strip()

MEMORY_GLOBAL_SUMMARY_USER_PROMPT = '''
# Individual Strategy Summaries
{individual_summaries}

# Past Experience (if any)
{previous_summary}

# Task
Analyze the strategies' implementation and actual performance, then produce an updated, self-contained Past Experience summary.

Focus on:
- Signal, filter, entry/exit, position sizing, risk control, and execution logic
- Returns, risk-adjusted performance, drawdown, turnover, costs, and stability
- Recurring successful and failed implementation patterns

Merge new findings with the existing Past Experience:
- Preserve conclusions still supported by evidence
- Strengthen repeated findings
- Revise or remove contradicted findings
- Deduplicate overlapping insights
- Do not simply append the new analysis

Distinguish repeated patterns from isolated or uncertain results.

# Output Format

## Successful Patterns
- Mechanism, relevant strategies, and supporting metrics

## Failure Patterns
- Harmful logic or implementation choices
- Relevant strategies, metric impact, and likely cause

# Note
Base insights ONLY on the ACTUAL individual strategy summaries and the existing Past Experience provided above — do not invent details.
Reference specific strategy names, metrics, and implementation details.
Do not make suggestions or recommend next steps. ONLY output one consolidated and deduplicated Past Experience summary.
'''.strip()
