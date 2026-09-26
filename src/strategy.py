import json


class Strategy:
    def __init__(
        self,
        id,
        code,
        strategy_dir,
        strategy_path=None,
        performance=None,
        parent_id=None,
    ):
        strategy_dir.mkdir(parents=True, exist_ok=True)
        if strategy_path is None:
            strategy_path = strategy_dir / "strategy.py"
        strategy_path.write_text(code)

        self.id = id
        self.code = code
        self.strategy_dir = strategy_dir
        self.strategy_path = strategy_path
        self.performance = performance
        self.parent_id = parent_id

    def save_lineage(self):
        """Persist the parent pointer so lineage survives a resume."""
        with open(self.strategy_dir / "lineage.json", "w") as f:
            json.dump({"id": self.id, "parent_id": self.parent_id}, f)

    @staticmethod
    def load_parent_id(strategy_dir):
        path = strategy_dir / "lineage.json"
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f).get("parent_id")
        except Exception:
            return None
