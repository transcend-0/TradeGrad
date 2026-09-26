"""Run TradeTGD to completion on one task, resuming if its checkpoint dir already has work.

Usage
-----
    python -m src.run_one --task cn_cross --max-steps 100
"""
import argparse
import json
import logging
import os
from pathlib import Path

from tqdm import tqdm

from .llm_client import LLMClient
from .loss import WindowStrategyReward, WindowTimeStrategyReward
from .optimizer import TradeTGD
from .strategy import Strategy
from .tasks import TASKS, TRAIN_START, TRAIN_END
from .util import Debugger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_loss(task, llm, worst_ratio):
    if task["mode"] == "time":
        return WindowTimeStrategyReward(
            llm, task["evaluate"], task["loss_kwargs"]["data_path"],
            TRAIN_START, TRAIN_END, worst_ratio=worst_ratio,
        )
    return WindowStrategyReward(
        llm, task["evaluate"],
        task["loss_kwargs"]["date2symbol_dir"], task["loss_kwargs"]["data_dir"],
        TRAIN_START, TRAIN_END, worst_ratio=worst_ratio,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--worst-ratio", type=float, default=0.3)
    # TradeTGD hyperparameters -- defaults match the class's own defaults.
    ap.add_argument("--sample-size", type=int, default=3)
    ap.add_argument("--large-revision-prob", type=float, default=0.7)
    ap.add_argument("--memory-window", type=int, default=5)
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--verbose", action="store_true",
                     help="write per-step traces, gradients and memory summaries to disk "
                          "(off by default -- nothing in this pipeline reads them back)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set; export it before launching a run.")

    task = TASKS[args.task]
    project_dir = Path(task["result_root"])

    llm = LLMClient(base_url=args.base_url, api_key=api_key, model_name=args.model)
    loss_train = build_loss(task, llm, args.worst_ratio)
    optimizer = TradeTGD(
        llm,
        sample_size=args.sample_size,
        large_revision_prob=args.large_revision_prob,
        memory_window=args.memory_window,
        verbose=args.verbose,
    )
    debugger = Debugger(llm, loss_train)

    logger.info(f"task={args.task} market={task['market']} train={TRAIN_START}~{TRAIN_END}")

    checkpoint_dir = project_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer.resume_database(checkpoint_dir)
    optimizer.resume_memory(checkpoint_dir)

    start_step = 0
    done = sorted((c for c in checkpoint_dir.glob("*") if c.name.isdigit()), key=lambda x: int(x.name))
    if done:
        start_step = int(done[-1].name)
        start_dir = checkpoint_dir / str(start_step)
        strategy = Strategy(
            id=start_step, code=(start_dir / "strategy.py").read_text(), strategy_dir=start_dir,
            performance=json.loads((start_dir / "metrics.json").read_text()),
        )
    else:
        strategy = Strategy(id=start_step, code=Path(task["initial"]).read_text(),
                             strategy_dir=checkpoint_dir / str(start_step))
        loss_train.forward(strategy)
        strategy = debugger.debug_strategy(strategy)

    for step in tqdm(range(start_step, args.max_steps), desc=args.task):
        grad = optimizer.backward(strategy, step)
        strategy_code = optimizer.safe_step(grad, fallback_code=strategy.code)

        strategy = Strategy(id=step + 1, code=strategy_code, strategy_dir=checkpoint_dir / str(step + 1))
        loss_train.forward(strategy)
        strategy = debugger.debug_strategy(strategy)

    if args.max_steps > start_step:
        optimizer.update_database(strategy)

    best = optimizer.best()
    if best is not None:
        best_dir = project_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / "strategy.py").write_text(best.code)
        (best_dir / "metrics.json").write_text(json.dumps(best.performance, indent=2))
        logger.info(f"=== training finished; best strategy is checkpoint {best.id} "
                    f"(combined_score={best.performance.get('combined_score')}) -> {best_dir} ===")
    else:
        logger.info("=== training finished; no scored strategy found ===")


if __name__ == "__main__":
    main()
