import pytest
from typing import List, Any
from abc import ABC, abstractmethod
from unittest.mock import patch

from lightrl import (
    Bandit,
    EpsilonGreedyBandit,
    EpsilonFirstBandit,
    EpsilonDecreasingBandit,
    UCB1Bandit,
    GreedyBanditWithHistory,
)


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
        self.arms = ["arm1", "arm2", "arm3"]
        self.bandit = EpsilonFirstBandit(self.arms, exploration_steps=2, epsilon=0.1)

    def test_initialization(self):
        bandit = EpsilonFirstBandit(self.arms, exploration_steps=5, epsilon=0.2)
        assert bandit.arms == self.arms, "Arms should be initialized correctly."
        assert (
            bandit.exploration_steps == 5
        ), "Exploration steps should be set correctly."
        assert bandit.epsilon == 0.2, "Epsilon should be set correctly."
        assert bandit.step == 0, "Initial step should be zero."

    def test_select_arm_pure_exploration(self):
        # During initial exploration steps
        self.bandit.step = 0
        with patch("random.randint", return_value=1):
            arm = self.bandit.select_arm()
            assert (
                arm == 1
            ), "During exploration steps, selected arm index should be random."

    def test_select_arm_epsilon_greedy_during_exploration(self):
        # During exploration phase affected by epsilon
        self.bandit.step = 1
        with patch("random.random", return_value=0.05):  # Lower than epsilon
            with patch("random.randint", return_value=2):
                arm = self.bandit.select_arm()
                assert (
                    arm == 2
                ), "By epsilon, selected arm index should be random during exploration."

    def test_select_arm_post_exploration_exploitation(self):
        # After exploration phase, epsilon is crucial
        self.bandit.step = 3
        self.bandit.q_values = [0.1, 0.8, 0.5]
        with patch("random.random", return_value=0.2):  # Greater than epsilon
            arm = self.bandit.select_arm()
            assert (
                arm == 1
            ), "After exploration, exploit: select the best arm based on q_values."

    def test_select_arm_post_exploration_exploration(self):
        # After exploration phase, random selection based on epsilon
        self.bandit.step = 3
        with patch("random.random", return_value=0.05):  # Less than epsilon
            with patch("random.randint", return_value=0):
                arm = self.bandit.select_arm()
                assert (
                    arm == 0
                ), "By epsilon, arm selection post-exploration should be random."


class TestEpsilonDecreasingBandit:
    def setup_method(self):
        """Set up a Bandit instance for testing."""
        self.arms = [0, 1, 2]
        self.bandit = EpsilonDecreasingBandit(
            arms=self.arms, initial_epsilon=1.0, limit_epsilon=0.1, half_decay_steps=100
        )

    def test_initialization(self):
        """Verify bandit initialization."""
        assert self.bandit.initial_epsilon == 1.0
        assert self.bandit.limit_epsilon == 0.1
        assert self.bandit.half_decay_steps == 100
        assert self.bandit.epsilon == 1.0
        assert self.bandit.step == 0

    def test_select_arm_explore(self):
        """Test that the bandit explores randomly based on epsilon probability."""
        with patch("random.random", return_value=0.5):  # Ensures we are exploring
            self.bandit.epsilon = 0.8
            arm_selected = self.bandit.select_arm()
            assert arm_selected in self.arms

    def test_select_arm_exploit(self):
        # TODO implement this test
        pass

    def test_update_epsilon(self):
        """Check that epsilon updates according to the decay model."""
        self.bandit.step = 50  # midway to half_decay
        self.bandit.update_epsilon()
        expected_epsilon = 0.1 + (1.0 - 0.1) * (0.5 ** (50 / 100))
        assert pytest.approx(self.bandit.epsilon, rel=1e-2) == expected_epsilon

    def test_epsilon_decay_to_limit(self):
        """Ensure epsilon decays towards the limit epsilon."""
        self.bandit.step = 1000  # beyond typical decay range
        self.bandit.update_epsilon()
        assert pytest.approx(self.bandit.epsilon, rel=1e-2) == self.bandit.limit_epsilon


class TestUCB1Bandit:
    def setup_method(self):
        """Set up a Bandit instance for testing."""
        self.arms = [0, 1, 2]
        self.bandit = UCB1Bandit(arms=self.arms)

    def test_initialization(self):
        """Verify bandit initialization."""
        assert self.bandit.total_count == 0
        assert self.bandit.q_values == [0.0, 0.0, 0.0]
        assert self.bandit.counts == [0, 0, 0]

    def test_select_first_arm(self):
        """Test that an unselected arm is chosen first."""
        arm = self.bandit.select_arm()
        assert (
            arm == 0
        )  # As no arm has been selected yet, it selects the first one with count 0

    def test_select_arm_ucb_calculation(self):
        """Test arm selection using UCB1 after some updates."""
        self.bandit.q_values = [0.5, 0.5, 0.5]
        self.bandit.counts = [1, 1, 1]
        self.bandit.total_count = 3

        with patch("math.log", return_value=1):  # Simplifying log for predictability
            arm = self.bandit.select_arm()
            assert arm == 0  # Due to tied UCB values, implementation select the first

    def test_update_arm_and_values(self):
        """Test that updating an arm modifies the right variables correctly."""
        self.bandit.update(0, 0.8)
        assert self.bandit.counts[0] == 1
        assert self.bandit.q_values[0] == 0.8
        assert self.bandit.total_count == 1

    def test_update_raises_value_error(self):
        """Test update method raises ValueError for invalid reward."""
        with pytest.raises(ValueError, match=r"Reward must be in the range \[0, 1\]."):
            self.bandit.update(0, 1.2)


class TestGreedyBanditWithHistory:
    def setup_method(self):
        """Set up the bandit instance for testing."""
        self.arms = [0, 1, 2]
        self.history_length = 5
        self.bandit = GreedyBanditWithHistory(
            arms=self.arms, history_length=self.history_length
        )

    def test_initialization(self):
        """Verify bandit initialization."""
        assert self.bandit.history_length == self.history_length
        assert len(self.bandit.history) == len(self.arms)
        assert all(len(history) == 0 for history in self.bandit.history)

    def test_select_arm_before_history_full(self):
        """Ensure random arm selection if any arm history is incomplete."""
        self.bandit.history[0] = [1] * (self.history_length - 1)
        arm = self.bandit.select_arm()
        assert arm == 0 or arm == 1 or arm == 2  # Any unfulfilled arm can be chosen

    def test_select_arm_after_history_full(self):
        """Test selection after fulfilling the history length."""
        self.bandit.q_values = [0.5, 0.8, 0.3]
        self.bandit.history = [[1] * self.history_length] * len(self.arms)
        arm = self.bandit.select_arm()
        assert arm == 1  # Since q_value for arm 1 is the highest

    def test_update_history(self):
        """Test that history maintains its bounded length and updates correctly."""
        for _ in range(self.history_length + 2):
            self.bandit.update(0, 1)

        assert len(self.bandit.history[0]) == self.history_length  # Should be capped
        assert self.bandit.q_values[0] == 1.0  # Average of history should be correct
        assert self.bandit.counts[0] == self.history_length

    def test_update_history_correct_mean(self):
        """Test updating alters Q-value of the arm correctly with mixed rewards."""
        rewards = [1.0, 0.8, 0.9, 0.7, 0.6]
        for reward in rewards:
            self.bandit.update(0, reward)

        expected_mean = sum(rewards) / len(rewards)
        assert pytest.approx(self.bandit.q_values[0], rel=1e-2) == expected_mean
        assert len(self.bandit.history[0]) == len(rewards)
