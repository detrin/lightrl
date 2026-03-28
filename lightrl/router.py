from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

from lightrl.bandits import Bandit, EpsilonGreedyBandit


class BanditRouter:
    def __init__(self, default_bandit_cls=EpsilonGreedyBandit, **default_kwargs):
        self._bandits: Dict[str, Bandit] = {}
        self._default_cls = default_bandit_cls
        self._default_kwargs = default_kwargs

    def register(self, name: str, bandit: Bandit) -> None:
        self._bandits[name] = bandit

    def _get_or_create(self, name: str, arms: Optional[list] = None) -> Bandit:
        if name not in self._bandits:
            if arms is None:
                raise ValueError(f"Bandit '{name}' not registered and no arms provided")
            self._bandits[name] = self._default_cls(arms=arms, **self._default_kwargs)
        return self._bandits[name]

    def select(self, name: str, arms: Optional[list] = None) -> int:
        bandit = self._get_or_create(name, arms)
        return bandit.select_arm()

    def select_arm_value(self, name: str, arms: Optional[list] = None):
        bandit = self._get_or_create(name, arms)
        idx = bandit.select_arm()
        return bandit.arms[idx]

    def update(self, name: str, arm_index: int, reward: float) -> None:
        self._bandits[name].update(arm_index, reward)

    def report(self, name: Optional[str] = None) -> None:
        targets = {name: self._bandits[name]} if name else self._bandits
        for n, b in targets.items():
            print(f"\n[{n}]")
            b.report()

    def save(self, path: Union[str, Path]) -> None:
        data = {}
        for name, bandit in self._bandits.items():
            data[name] = {"class": bandit.__class__.__name__, "state": bandit.__dict__.copy()}
        Path(path).write_text(json.dumps(data, default=str))

    @classmethod
    def load(cls, path: Union[str, Path], default_bandit_cls=EpsilonGreedyBandit, **default_kwargs):
        from lightrl.bandits import _all_bandit_classes

        registry = {c.__name__: c for c in _all_bandit_classes()}
        data = json.loads(Path(path).read_text())
        router = cls(default_bandit_cls=default_bandit_cls, **default_kwargs)
        for name, entry in data.items():
            klass = registry[entry["class"]]
            obj = object.__new__(klass)
            obj.__dict__.update(entry["state"])
            router._bandits[name] = obj
        return router
