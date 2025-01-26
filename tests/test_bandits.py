import pytest
from typing import List, Any
from abc import ABC, abstractmethod
from unittest.mock import patch

from lightrl import Bandit, EpsilonGreedyBandit, EpsilonFirstBandit


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


class TestEpsilonGreedyBandit:

    def setup_method(self):
        self.arms = ["arm1", "arm2", "arm3"]
        self.bandit = EpsilonGreedyBandit(self.arms, epsilon=0.1)

    def test_select_arm_exploration(self):
        # Force exploration by setting random.random to be less than epsilon
        with patch("random.random", return_value=0.05):
            arm = self.bandit.select_arm()
            assert (
                0 <= arm < len(self.arms)
            ), "Selected arm index should be within range of arms."

    def test_select_arm_exploitation(self):
        # Set q_values to known values
        self.bandit.q_values = [0.1, 0.5, 0.2]  # arm1 and arm3 are worse than arm2
        with patch("random.random", return_value=0.2):
            arm = self.bandit.select_arm()
            assert arm == 1, "Arm2 should be selected as it has the highest q-value."

    def test_select_arm_full_exploration(self):
        # Test with epsilon of 1 for full exploration
        self.bandit.epsilon = 1
        with patch("random.randint", return_value=2):
            arm = self.bandit.select_arm()
            assert (
                arm == 2
            ), "Selected arm should match random choice due to full exploration."

    def test_select_arm_full_exploitation(self):
        # Test with epsilon of 0 for full exploitation
        self.bandit.epsilon = 0
        self.bandit.q_values = [
            0.1,
            0.5,
            0.5,
        ]  # arm2 and arm3 are the best with equal q-values
        with patch(
            "random.choice", return_value=2
        ):  # Prefer the second best arm due to tie-breaking
            arm = self.bandit.select_arm()
            assert arm in [
                1,
                2,
            ], "Arm2 or Arm3 should be selected as they have the highest q-values."



class TestEpsilonFirstBandit:

    def setup_method(self):
        self.arms = ['arm1', 'arm2', 'arm3']
        self.bandit = EpsilonFirstBandit(self.arms, exploration_steps=2, epsilon=0.1)

    def test_initialization(self):
        bandit = EpsilonFirstBandit(self.arms, exploration_steps=5, epsilon=0.2)
        assert bandit.arms == self.arms, "Arms should be initialized correctly."
        assert bandit.exploration_steps == 5, "Exploration steps should be set correctly."
        assert bandit.epsilon == 0.2, "Epsilon should be set correctly."
        assert bandit.step == 0, "Initial step should be zero."

    def test_select_arm_pure_exploration(self):
        # During initial exploration steps
        self.bandit.step = 0
        with patch('random.randint', return_value=1):
            arm = self.bandit.select_arm()
            assert arm == 1, "During exploration steps, selected arm index should be random."

    def test_select_arm_epsilon_greedy_during_exploration(self):
        # During exploration phase affected by epsilon
        self.bandit.step = 1
        with patch('random.random', return_value=0.05):  # Lower than epsilon
            with patch('random.randint', return_value=2):
                arm = self.bandit.select_arm()
                assert arm == 2, "By epsilon, selected arm index should be random during exploration."

    def test_select_arm_post_exploration_exploitation(self):
        # After exploration phase, epsilon is crucial
        self.bandit.step = 3
        self.bandit.q_values = [0.1, 0.8, 0.5]
        with patch('random.random', return_value=0.2):  # Greater than epsilon
            arm = self.bandit.select_arm()
            assert arm == 1, "After exploration, exploit: select the best arm based on q_values."

    def test_select_arm_post_exploration_exploration(self):
        # After exploration phase, random selection based on epsilon
        self.bandit.step = 3
        with patch('random.random', return_value=0.05):  # Less than epsilon
            with patch('random.randint', return_value=0):
                arm = self.bandit.select_arm()
                assert arm == 0, "By epsilon, arm selection post-exploration should be random."
