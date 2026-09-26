import re
import time
import shutil

from .strategy import Strategy

import logging

logger = logging.getLogger(__name__)



def retry_on_exception(max_retries=5, exceptions=(Exception,), delay=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retrying {retries + 1}/{max_retries}... {type(e).__name__}: {e}.")
                    retries += 1
                    time.sleep(delay)
            raise Exception(f"Failed after {max_retries} retries.")
        return wrapper
    return decorator

def parse_code(code: str, language: str = 'python') -> str:
    '''
    Extract code from a string, removing any markdown formatting.
    '''
    match = re.search(rf'```{language}\n(.*?)\n```', code, flags=re.DOTALL)
    if match:
        code = match.group(1)
    else:
        raise ValueError('No code block found in the response.')
    if not re.search(r'class \w+Strategy\(Strategy\):', code):
        raise ValueError('Modified code must contain a strategy class definition `class <Name>Strategy(Strategy):`.')
    if '### EDIT START' not in code or '### EDIT END' not in code:
        raise ValueError('Modified code must contain `### EDIT START` and `### EDIT END` markers.')
    return code.strip()

def parse_diffs(diff: str) -> list:
    '''
    Parse ALL SEARCH/REPLACE blocks in a response. The MID/LARGE step prompts ask for several
    blocks; parsing only the first silently dropped the rest of the edit.
    '''
    pattern = r'<{3,}\s*SEARCH\n(.*?)\n={3,}\n(.*?)\n>{3,}\s*REPLACE'
    matches = re.findall(pattern, diff, flags=re.DOTALL)
    if not matches:
        raise ValueError('Invalid diff format. Expected SEARCH/REPLACE format.')
    return [(o.strip(), r.strip()) for o, r in matches]


def apply_diffs(code: str, blocks: list) -> str:
    """Apply every SEARCH/REPLACE block that matches; raise only if none of them do.

    Requiring all blocks to match made the step brittle: the model has to reproduce each SEARCH
    section verbatim, so the chance that every one of k blocks matches falls as k grows, and on the
    cross-sectional task -- where the strategies are longer -- a run died at step 32 after five
    consecutive all-or-nothing failures.

    Dropping the whole edit because one block out of three was mistranscribed also throws away two
    good edits. Partial application keeps them, and a step where nothing matched still raises so the
    retry can ask again rather than silently returning the parent unchanged.
    """
    applied = 0
    missed = []
    for original, replacement in blocks:
        if original in code:
            code = code.replace(original, replacement, 1)
            applied += 1
        else:
            missed.append(original.splitlines()[0][:60] if original else '')
    if applied == 0:
        raise ValueError(
            'No SEARCH block matched the strategy code; retrying rather than returning the parent '
            'unchanged.'
        )
    if missed:
        logger.info(
            f'Applied {applied}/{len(blocks)} SEARCH blocks; {len(missed)} did not match '
            f'(first lines: {missed})'
        )
    return code


def parse_diff(diff: str) -> tuple:
    '''
    Parse a diff in SEARCH/REPLACE format and return the modified code.
    '''
    pattern = r'<{3,}\s*SEARCH\n(.*?)\n={3,}\n(.*?)\n>{3,}\s*REPLACE'
    match = re.search(pattern, diff, flags=re.DOTALL)
    if match:
        original, replacement = match.groups()
    else:
        raise ValueError('Invalid diff format. Expected SEARCH/REPLACE format.')
    return original.strip(), replacement.strip()


class Debugger:
    def __init__(self, llm, loss, max_debug_times=5):
        self.llm = llm
        self.loss = loss
        self.max_debug_times = max_debug_times

    @retry_on_exception()
    def debug_code(self, strategy, error):
        user_prompt = f'''
Debug the code based on the error, and output a corrected version of full code within ```python ... ``` code block.
# Code
```python
{strategy.code}
```
# Error
```
{error}
```
# Note
- You can only modify the code within `### EDIT START` and `### EDIT END` markers. Do not change anything outside these markers.
- Prioritize vectorized operations and efficient data handling when making improvements.
'''.strip()
        response = self.llm.query(user_prompt)
        code = parse_code(response)
        return Strategy(
            id=strategy.id,
            code=code,
            strategy_dir=strategy.strategy_dir,
        )

    def debug_strategy(self, strategy):
        for i in range(self.max_debug_times):
            error = strategy.performance.get('error')
            if not error:
                return strategy
            logger.info(f"Debugging strategy {strategy.id}, attempt {i+1}/{self.max_debug_times}...")
            shutil.rmtree(strategy.strategy_dir, ignore_errors=True)
            try:
                strategy = self.debug_code(strategy, error)
            except Exception as e:
                # The repair model can fail every retry -- it has been observed to keep returning
                # code without the `### EDIT` markers late into a run -- and letting that propagate
                # would kill a run that was mostly done. A program that cannot be repaired is a legitimate
                # search outcome, not a reason to stop: keep it, let it score 0, carry on. The
                # rmtree above removed its directory, so rebuild it first or the checkpoint would
                # be left missing.
                logger.info(f"Debugger gave up on strategy {strategy.id} ({e}); keeping it unrepaired.")
                strategy = Strategy(id=strategy.id, code=strategy.code,
                                    strategy_dir=strategy.strategy_dir,
                                    performance=strategy.performance,
                                    parent_id=getattr(strategy, 'parent_id', None))
                self.loss.forward(strategy)
                return strategy
            performance = self.loss.forward(strategy)
        return strategy