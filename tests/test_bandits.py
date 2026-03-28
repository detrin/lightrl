from unittest.mock import patch

import pytest

from lightrl import (
    Bandit,
    EpsilonDecreasingBandit,
    EpsilonFirstBandit,
    EpsilonGreedyBandit,
    GreedyBanditWithHistory,
    UCB1Bandit,
)


class TestBandit:
    def setup_method(self):
        class ConcreteBandit(Bandit):
            def select_arm(self) -> int:
                return 0

        self.bandit = ConcreteBandit(arms=[1, 2, 3])

    def test_initialization(self):
        assert self.bandit.arms == [1, 2, 3]
        assert self.bandit.q_values == [0.0, 0.0, 0.0]
        assert self.bandit.counts == [0, 0, 0]

    def test_update(self):
        self.bandit.update(0, 1.0)
        assert self.bandit.q_values[0] == 1.0
        assert self.bandit.counts[0] == 1
        self.bandit.update(0, 0.5)
        assert self.bandit.q_values[0] == 0.75
        assert self.bandit.counts[0] == 2

    def test_report(self, capsys):
        self.bandit.update(0, 1.0)
        self.bandit.update(1, 0.5)
        self.bandit.report()
        out = capsys.readouterr().out
        assert "num_tasks=1: avg_reward=1.00000, count=1" in out
        assert "num_tasks=2: avg_reward=0.50000, count=1" in out
        assert "num_tasks=3: avg_reward=0.00000, count=0" in out

    def test_repr(self):
        assert repr(self.bandit) == "ConcreteBandit(arms=[1, 2, 3])"


class TestEpsilonGreedyBandit:
    def setup_method(self):
        self.bandit = EpsilonGreedyBandit(["a", "b", "c"], epsilon=0.1)

    def test_exploration(self):
        with patch("random.random", return_value=0.05):
            assert 0 <= self.bandit.select_arm() < 3

    def test_exploitation(self):
        self.bandit.q_values = [0.1, 0.5, 0.2]
        with patch("random.random", return_value=0.2):
            assert self.bandit.select_arm() == 1

    def test_full_exploration(self):
        self.bandit.epsilon = 1
        with patch("random.randint", return_value=2):
            assert self.bandit.select_arm() == 2

    def test_full_exploitation(self):
        self.bandit.epsilon = 0
        self.bandit.q_values = [0.1, 0.5, 0.5]
        with patch("random.choice", return_value=2):
            assert self.bandit.select_arm() in [1, 2]


class TestEpsilonFirstBandit:
    def setup_method(self):
        self.bandit = EpsilonFirstBandit(["a", "b", "c"], exploration_steps=2, epsilon=0.1)

    def test_initialization(self):
        b = EpsilonFirstBandit(["a", "b"], exploration_steps=5, epsilon=0.2)
        assert b.exploration_steps == 5
        assert b.epsilon == 0.2
        assert b.step == 0

    def test_pure_exploration(self):
        self.bandit.step = 0
        with patch("random.randint", return_value=1):
            assert self.bandit.select_arm() == 1

    def test_epsilon_during_exploration(self):
        self.bandit.step = 1
        with patch("random.random", return_value=0.05), patch("random.randint", return_value=2):
            assert self.bandit.select_arm() == 2

    def test_post_exploration_exploit(self):
        self.bandit.step = 3
        self.bandit.q_values = [0.1, 0.8, 0.5]
        with patch("random.random", return_value=0.2):
            assert self.bandit.select_arm() == 1

    def test_post_exploration_explore(self):
        self.bandit.step = 3
        with patch("random.random", return_value=0.05), patch("random.randint", return_value=0):
            assert self.bandit.select_arm() == 0


class TestEpsilonDecreasingBandit:
    def setup_method(self):
        self.bandit = EpsilonDecreasingBandit(
            arms=[0, 1, 2], initial_epsilon=1.0, limit_epsilon=0.1, half_decay_steps=100
        )

    def test_initialization(self):
        assert self.bandit.initial_epsilon == 1.0
        assert self.bandit.limit_epsilon == 0.1
        assert self.bandit.half_decay_steps == 100
        assert self.bandit.step == 0

    def test_exploration(self):
        with patch("random.random", return_value=0.5):
            self.bandit.epsilon = 0.8
            assert self.bandit.select_arm() in [0, 1, 2]

    def test_update_epsilon(self):
        self.bandit.step = 50
        self.bandit._update_epsilon()
        expected = 0.1 + 0.9 * (0.5 ** (50 / 100))
        assert pytest.approx(self.bandit.epsilon, rel=1e-2) == expected

    def test_epsilon_decay_to_limit(self):
        self.bandit.step = 1000
        self.bandit._update_epsilon()
        assert pytest.approx(self.bandit.epsilon, rel=1e-2) == self.bandit.limit_epsilon


class TestUCB1Bandit:
    def setup_method(self):
        self.bandit = UCB1Bandit(arms=[0, 1, 2])

    def test_initialization(self):
        assert self.bandit.total_count == 0
        assert self.bandit.q_values == [0.0, 0.0, 0.0]

    def test_select_unvisited(self):
        assert self.bandit.select_arm() == 0

    def test_ucb_calculation(self):
        self.bandit.q_values = [0.5, 0.5, 0.5]
        self.bandit.counts = [1, 1, 1]
        self.bandit.total_count = 3
        with patch("math.log", return_value=1):
            assert self.bandit.select_arm() == 0

    def test_update(self):
        self.bandit.update(0, 0.8)
        assert self.bandit.counts[0] == 1
        assert self.bandit.q_values[0] == 0.8
        assert self.bandit.total_count == 1

    def test_update_invalid_reward(self):
        with pytest.raises(ValueError, match=r"Reward must be in the range \[0, 1\]."):
            self.bandit.update(0, 1.2)


class TestGreedyBanditWithHistory:
    def setup_method(self):
        self.bandit = GreedyBanditWithHistory(arms=[0, 1, 2], history_length=5)

    def test_initialization(self):
        assert self.bandit.history_length == 5
        assert all(len(h) == 0 for h in self.bandit.history)

    def test_select_before_full(self):
        self.bandit.history[0] = [1.0] * 4
        assert self.bandit.select_arm() in [0, 1, 2]

    def test_select_after_full(self):
        self.bandit.q_values = [0.5, 0.8, 0.3]
        self.bandit.history = [[1.0] * 5 for _ in range(3)]
        assert self.bandit.select_arm() == 1

    def test_history_bounded(self):
        for _ in range(7):
            self.bandit.update(0, 1.0)
        assert len(self.bandit.history[0]) == 5
        assert self.bandit.q_values[0] == 1.0

    def test_history_mean(self):
        rewards = [1.0, 0.8, 0.9, 0.7, 0.6]
        for r in rewards:
            self.bandit.update(0, r)
        assert pytest.approx(self.bandit.q_values[0], rel=1e-2) == sum(rewards) / len(rewards)
