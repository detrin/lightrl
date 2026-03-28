import pytest

from lightrl import EpsilonGreedyBandit


class TestPriors:
    def test_priors_set_initial_q_values(self):
        b = EpsilonGreedyBandit(arms=["a", "b", "c"], priors=[0.3, 0.7, 0.9], epsilon=0.0)
        assert b.q_values == [0.3, 0.7, 0.9]

    def test_priors_influence_first_selection(self):
        b = EpsilonGreedyBandit(arms=["a", "b", "c"], priors=[0.1, 0.1, 0.9], epsilon=0.0)
        assert b.select_arm() == 2

    def test_priors_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="priors length"):
            EpsilonGreedyBandit(arms=["a", "b", "c"], priors=[0.5, 0.8])


class TestEMADecay:
    def test_ema_updates_with_alpha(self):
        b = EpsilonGreedyBandit(arms=["a", "b"], ema_alpha=0.5)
        b.update(0, 1.0)
        assert b.q_values[0] == 0.5
        b.update(0, 1.0)
        assert b.q_values[0] == 0.75

    def test_ema_zero_falls_back_to_average(self):
        b = EpsilonGreedyBandit(arms=["a", "b"], ema_alpha=0.0)
        b.update(0, 1.0)
        b.update(0, 0.0)
        assert b.q_values[0] == 0.5

    def test_ema_adapts_to_change(self):
        b = EpsilonGreedyBandit(arms=["a", "b"], ema_alpha=0.3)
        for _ in range(100):
            b.update(0, 1.0)
        assert b.q_values[0] > 0.95
        for _ in range(100):
            b.update(0, 0.0)
        assert b.q_values[0] < 0.1
