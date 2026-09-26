import json
import logging
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .prompt import *
from .strategy import Strategy
from .llm_client import LLMClient
from .util import retry_on_exception, parse_code, parse_diffs, apply_diffs

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def step(self, *args, **kwargs):
        '''
        Perform one optimization step.
        '''
        pass

    def safe_step(self, feedback, fallback_code=None):
        '''Run one step, and on total generation failure return the base unchanged.

        `step` is wrapped in retry_on_exception, which raises once the retries are spent, and the
        main loop has no handler -- so five consecutive malformed generations end the run. A step
        that produces nothing is a wasted evaluation; a step that raises costs every remaining step
        in the run.
        '''
        try:
            return self.step(feedback)
        except Exception as e:
            try:
                base = self._select_base()
            except Exception:
                base = None
            code = fallback_code if base is None else base.code
            logger.info(f'Generation failed for this step ({e}); keeping the base program unchanged.')
            return code


class TradeTGD(Optimizer):
    """Textual gradient descent over trading strategies: uniform base selection, rank-based
    reference sampling, a running memory summary, and a stochastic small/large revision size.

    Each step:
      1. sample a reference set S_k from the score-ranked archive (power-law / uniform / top-k);
      2. write a textual gradient against S_k, folding in the running memory summary of the last
         `memory_window` strategies (refreshed on that cadence);
      3. pick the base program *uniformly* from S_k, rather than always taking the top scorer --
         taking the max every time lets one high scorer dominate every step and freezes the search
         around it;
      4. revise the base as either a small, localized SEARCH/REPLACE edit or a full rewrite,
         chosen stochastically via `large_revision_prob`;
      5. backtest the revised program and add it to the archive.

    Every scored strategy is kept in `self.database` (the archive A), sorted by score, so the
    optimized strategy pi* = argmax_{(pi, B, J) in A} J is always `self.database[0]` -- see `best()`.

    `verbose=True` additionally writes, per step: the gradient and its prompt, the memory
    summaries, a full generation trace (both prompts, the raw response, the resulting code), and
    each checkpoint's parent id (`lineage.json`). None of it is read by this pipeline -- it exists
    for offline inspection of a run -- so it is off by default.
    """

    def __init__(
        self,
        llm: LLMClient,
        sample_size: int = 3,
        sample_method: str = 'power_law',
        large_revision_prob: float = 0.7,
        memory_window: int = 5,
        verbose: bool = False,
    ):
        super().__init__(llm)
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.large_revision_prob = large_revision_prob
        self.memory_window = memory_window
        self.verbose = verbose

        self.database = []
        self.memory_storage = []
        self.memory_summary = ''

        self.pending_parent_id = None

    # Default so a trace write or a step before the first backward() cannot raise.
    current_revision_mode = 'small'

    # ------------------------------------------------------------------ database / resume

    def resume_database(self, checkpoint_dir: Path):
        if any(checkpoint_dir.iterdir()):
            checkpoints = sorted(checkpoint_dir.glob('*'), key=lambda x: int(x.name))
            for checkpoint in checkpoints:
                strategy_path = checkpoint / 'strategy.py'
                if strategy_path.exists() and Path(checkpoint / 'metrics.json').exists():
                    with open(checkpoint / 'metrics.json', 'r') as f:
                        performance = json.load(f)
                    strategy = Strategy(
                        id=int(checkpoint.name),
                        code=strategy_path.read_text(),
                        strategy_dir=checkpoint,
                        performance=performance,
                    )
                    self.update_database(strategy)

    def resume_memory(self, checkpoint_dir: Path):
        """Restore the accumulated memory and the recent-strategy window after a resume.

        `resume_database` rebuilds the population but not the memory, so a resumed run without this
        would restart from an empty "Past Experience" and rebuild its window from scratch. The
        latest `memory_global_summary.txt` on disk is the memory as of the last update, and the
        most recent checkpoints reconstruct the FIFO window. Both only exist if the run was started
        with `verbose=True`; otherwise this is a no-op and memory rebuilds from scratch.
        """
        summaries = sorted(
            (p for p in checkpoint_dir.glob('*/memory_global_summary.txt')
             if p.parent.name.isdigit()),
            key=lambda p: int(p.parent.name),
        )
        if summaries:
            self.memory_summary = summaries[-1].read_text().strip()
            logger.info(
                f'Resumed memory from {summaries[-1]} ({len(self.memory_summary.split())} words)'
            )

        ids = sorted((int(p.name) for p in checkpoint_dir.glob('*') if p.name.isdigit()))
        by_id = {s.id: s for s in self.database}
        self.memory_storage = [by_id[i] for i in ids[-self.memory_window:] if i in by_id]
        logger.info(f'Resumed memory window with {len(self.memory_storage)} strategies')

    def update_database(self, strategy: Strategy):
        if self.verbose:
            # Record the strategy this one was actually derived from (lineage.json next to each
            # checkpoint), for offline analysis of a run. Nothing in this pipeline reads it back,
            # so it only costs a file write when verbose is on.
            if getattr(strategy, 'parent_id', None) is None:
                strategy.parent_id = self.pending_parent_id
            if strategy.parent_id is None:
                try:
                    strategy.parent_id = Strategy.load_parent_id(strategy.strategy_dir)
                except Exception:
                    pass
            if strategy.parent_id is not None:
                try:
                    strategy.save_lineage()
                except Exception:
                    pass
        # Replace rather than append if this id is already archived -- resuming a run reloads
        # every done checkpoint via resume_database() and then processes the last one again (it
        # is both the tail of the prior run and the base for this run's first step), which would
        # otherwise double-count it in the archive and skew rank-based sampling towards it.
        self.database = [s for s in self.database if s.id != strategy.id]
        self.database.append(strategy)
        self.database.sort(
            key=lambda s: s.performance.get('combined_score', float('-inf')),
            reverse=True,
        )

    def best(self) -> Strategy | None:
        """pi* = argmax_{(pi, B, J) in A} J: the highest-scoring strategy found so far.

        The archive is kept sorted by score in `update_database`, so this is always its head.
        """
        return self.database[0] if self.database else None

    def update_memory_storage(self, strategy: Strategy):
        self.memory_storage.append(strategy)
        if len(self.memory_storage) > self.memory_window:
            self.memory_storage.pop(0)

    # ------------------------------------------------------------------ sampling

    def _sample_with_powerlaw(self, items: list, size: int, alpha: float = 1.0) -> list:
        '''Samples items using a power-law distribution based on their rank.

        Args:
            items (list): List of items to sample from (order implies rank, best first).
            size (int): Number of items to sample.
            alpha (float, optional): Power law exponent. Defaults to 1.0.
                - alpha = 0: uniform sampling.
                - alpha > 0: items earlier in the list (higher rank) are sampled more.
                - alpha < 0: items later in the list (lower rank) are sampled more.
        '''
        probs = np.array([(i + 1) ** (-alpha) for i in range(len(items))])
        if np.sum(probs) == 0:
            probs = np.ones(len(items))
        probs = probs / probs.sum()
        indices = np.random.choice(len(items), size=size, p=probs, replace=False)
        return [items[i] for i in indices]

    def sample_strategies(self, sample_method, sample_size):
        n = min(sample_size, len(self.database))
        if sample_method == 'uniform':
            return random.sample(self.database, n)
        elif sample_method == 'topk':
            return self.database[:n]
        elif sample_method == 'power_law':
            return self._sample_with_powerlaw(self.database, size=n, alpha=1.0)
        else:
            raise ValueError(f'Unknown sample_method: {sample_method}')

    def _select_base(self):
        """The program a step is taken from: uniform draw from the sampled reference pool.

        With plain rank-based sampling the references tend to cluster around one high scorer, so
        drawing the base uniformly among them -- instead of always taking the max -- is what keeps
        the search from freezing around whichever program reached the top first.
        """
        pool = getattr(self, 'sampled_strategies', None)
        if not pool:
            return max(self.database, key=lambda s: s.performance.get('combined_score', float('-inf')))
        base = pool[int(np.random.randint(len(pool)))]
        if self.verbose:
            logger.info(
                f'Uniform base {base.id} (score {base.performance.get("combined_score")}) '
                f'from sampled {[s.id for s in pool]}'
            )
        return base

    # ------------------------------------------------------------------ memory summary

    def summarize_strategy(self, strategy: Strategy) -> str:
        return self.llm.query(
            MEMORY_INDIVIDUAL_SUMMARY_USER_PROMPT.format(
                strategy_code=strategy.code,
                performance=strategy.performance,
            ),
            MEMORY_INDIVIDUAL_SUMMARY_SYSTEM_PROMPT,
        ).strip()

    def update_memory_summary(self):
        logger.info('Updating memory summary based on recent strategies...')
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.summarize_strategy, strategy): strategy
                for strategy in self.memory_storage
            }
            summaries = {strategy: future.result() for future, strategy in futures.items()}

        self.memory_summary = self.llm.query(
            MEMORY_GLOBAL_SUMMARY_USER_PROMPT.format(
                individual_summaries='\n\n'.join(summaries.values()),
                previous_summary=self.memory_summary,
            ),
            MEMORY_GLOBAL_SUMMARY_SYSTEM_PROMPT,
        ).strip()

        if self.verbose:
            for strategy, summary in summaries.items():
                (strategy.strategy_dir / 'memory_summary.txt').write_text(summary)
            if self.memory_storage:
                (self.memory_storage[-1].strategy_dir / 'memory_global_summary.txt').write_text(
                    self.memory_summary
                )

    # ------------------------------------------------------------------ backward / step

    @retry_on_exception()
    def backward(self, strategy, step):
        self.update_memory_storage(strategy)
        if step % self.memory_window == 0 and step != 0:
            self.update_memory_summary()

        self.update_database(strategy)

        sampled_strategies = self.sample_strategies(self.sample_method, self.sample_size)
        if not sampled_strategies:
            return 'No historical strategies found.'
        self.sampled_strategies = sampled_strategies

        if np.random.random() < self.large_revision_prob:
            self.current_revision_mode = 'large'
            prompt_template = GRAD_EXPLOIT_USER_PROMPT_LARGE_REVISION
            logger.info('Using a LARGE revision...')
        else:
            self.current_revision_mode = 'small'
            prompt_template = GRAD_EXPLOIT_USER_PROMPT_SMALL_REVISION
            logger.info('Using a SMALL revision...')

        user_prompt = prompt_template.format(
            reference_strategies_with_performance='\n\n'.join(
                SINGLE_STRATEGY_PERFORMANCE_TEMPLATE.format(
                    id=str(s.id), strategy_code=s.code, performance=str(s.performance)
                )
                for s in sampled_strategies
            ),
        )
        if self.memory_summary:
            user_prompt += f'\n\n# Past Experience\n{self.memory_summary}'

        grad = self.llm.query(user_prompt, REWARD_SYSTEM_PROMPT)
        if self.verbose:
            (strategy.strategy_dir / 'grad.txt').write_text(grad)
            (strategy.strategy_dir / 'grad_prompt.txt').write_text(user_prompt)
        return grad

    @retry_on_exception()
    def step(self, feedback: str):
        base_strategy = self._select_base()
        self.pending_parent_id = base_strategy.id

        if self.current_revision_mode == 'small':
            system_prompt = OPTIMIZER_SYSTEM_PROMPT_SMALL_REVISION
            user_prompt = OPTIMIZER_USER_PROMPT_SMALL_REVISION.format(strategy_code=base_strategy.code, feedback=feedback)
            response = self.llm.query(user_prompt, system_prompt)
            try:
                code = apply_diffs(base_strategy.code, parse_diffs(response))
            except ValueError as e:
                # `apply_diffs` raises when no SEARCH block matches, so a step cannot silently
                # return the parent unchanged. Rewriting the whole block instead keeps the step
                # productive, along the same generation path a large revision would take.
                logger.info(f'{e} Falling back to a full rewrite for this step.')
                system_prompt = OPTIMIZER_SYSTEM_PROMPT
                user_prompt = OPTIMIZER_USER_PROMPT.format(strategy_code=base_strategy.code, feedback=feedback)
                response = self.llm.query(user_prompt, system_prompt)
                code = parse_code(response)
        else:
            system_prompt = OPTIMIZER_SYSTEM_PROMPT
            user_prompt = OPTIMIZER_USER_PROMPT.format(strategy_code=base_strategy.code, feedback=feedback)
            response = self.llm.query(user_prompt, system_prompt)
            code = parse_code(response)

        if self.verbose:
            # Full trace of the generation stage, written next to the strategy this step was based
            # on so a run can be replayed end to end: base, both prompts, the raw response and the
            # code that came out of it. A base can be expanded several times, so this accumulates
            # one numbered entry per generation from it; `output_sha` links each entry to the
            # checkpoint whose code it produced (the child's id does not exist yet here).
            import hashlib
            trace = base_strategy.strategy_dir / 'step_trace'
            trace.mkdir(exist_ok=True)
            n = len(list(trace.glob('*_meta.json')))
            (trace / f'{n}_meta.json').write_text(json.dumps({
                'revision_mode': self.current_revision_mode,
                'base_id': base_strategy.id,
                'base_score': (base_strategy.performance or {}).get('combined_score'),
                'sampled_ids': [s.id for s in getattr(self, 'sampled_strategies', [])],
                'output_sha': hashlib.sha1(code.encode()).hexdigest(),
            }, indent=2))
            (trace / f'{n}_system_prompt.txt').write_text(system_prompt)
            (trace / f'{n}_user_prompt.txt').write_text(user_prompt)
            (trace / f'{n}_response.txt').write_text(response)
            (trace / f'{n}_output_code.py').write_text(code)
        return code
