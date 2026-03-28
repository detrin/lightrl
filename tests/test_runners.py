from unittest.mock import MagicMock

import pytest

from lightrl import two_state_time_dependent_process


class TestTwoStateTimeDependentProcess:
    @pytest.fixture
    def mock_bandit(self):
        b = MagicMock()
        b.select_arm.side_effect = [0, 1, 0]
        b.arms = [(1, 2), (3, 4)]
        return b

    def fun(self, *args):
        return (10, 0) if args == (1, 2) else (5, 5)

    def test_missing_waiting_args(self, mock_bandit):
        with pytest.raises(ValueError):
            two_state_time_dependent_process(
                bandit=mock_bandit,
                fun=self.fun,
                waiting_args=None,
                max_steps=3,
                default_wait_time=0.01,
                extra_wait_time=0.01,
            )

    def test_alive_flow(self, mock_bandit):
        two_state_time_dependent_process(
            bandit=mock_bandit,
            fun=self.fun,
            waiting_args=(3, 4),
            max_steps=3,
            default_wait_time=0.01,
            extra_wait_time=0.01,
        )
        assert mock_bandit.update.call_count > 0

    def test_state_transition(self, mock_bandit):
        two_state_time_dependent_process(
            bandit=mock_bandit,
            fun=lambda *a: (1, 9),
            failure_threshold=0.5,
            waiting_args=(1, 2),
            max_steps=3,
            default_wait_time=0.01,
            extra_wait_time=0.01,
            verbose=True,
        )
        assert mock_bandit.select_arm.call_count > 0
        assert mock_bandit.update.call_count == 0
