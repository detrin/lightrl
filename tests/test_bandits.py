import pytest
from typing import List, Any
from abc import ABC, abstractmethod

from lightrl import Bandit

class TestBandit:
    
    def setup_method(self):
        class ConcreteBandit(Bandit):
            def select_arm(self) -> int:
                # A simple implementation for testing that always selects the first arm
                return 0
    
        self.bandit = ConcreteBandit(arms=[1, 2, 3])

    def test_bandit_initialization(self):
        # Test the initialization of the bandit
        assert self.bandit.arms == [1, 2, 3]
        assert self.bandit.q_values == [0.0, 0.0, 0.0]
        assert self.bandit.counts == [0, 0, 0]

    def test_bandit_update(self):
        # Test the update method
        self.bandit.update(arm_index=0, reward=1.0)
        assert self.bandit.q_values[0] == 1.0
        assert self.bandit.counts[0] == 1

        self.bandit.update(arm_index=0, reward=0.5)
        assert self.bandit.q_values[0] == 0.75  # Average of 1.0 and 0.5
        assert self.bandit.counts[0] == 2

    def test_bandit_report(self, capsys):
        # Test the report method by capturing the printed output
        self.bandit.update(arm_index=0, reward=1.0)
        self.bandit.update(arm_index=1, reward=0.5)
        self.bandit.report()

        captured = capsys.readouterr()
        expected_output = (
            "Q-values per arm:\n"
            "  num_tasks=1: avg_reward=1.00000, count=1\n"
            "  num_tasks=2: avg_reward=0.50000, count=1\n"
            "  num_tasks=3: avg_reward=0.00000, count=0\n"
        )
        assert captured.out == expected_output

    def test_bandit_repr(self):
        # Test the __repr__ method
        assert repr(self.bandit) == "ConcreteBandit(arms=[1, 2, 3])"