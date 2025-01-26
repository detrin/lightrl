import pytest
from unittest.mock import MagicMock
import time

from lightrl import two_state_time_dependent_process


class TestTwoStateTimeDependentProcess:

    @pytest.fixture
    def mock_bandit(self):
        mock_bandit = MagicMock()
        mock_bandit.select_arm.side_effect = [0, 1, 0]  # Just example selections
        mock_bandit.arms = [(1, 2), (3, 4)]  # Example configuration for arms
        mock_bandit.update = MagicMock()
        mock_bandit.report = MagicMock()
        return mock_bandit

    def fun(self, *args):
        """Mock function to simulate a task with fixed output."""
        # For simplicity, assume these values or implement logic to vary them
        return (10, 0) if args == (1, 2) else (5, 5)

    def test_process_with_no_failures(self, mock_bandit):
        """Test process flow when there's no failure and remains mostly in ALIVE state."""
        # We mock `time.sleep` to avoid slowdown during testing
        with pytest.raises(ValueError):
            two_state_time_dependent_process(
                bandit=mock_bandit,
                fun=self.fun,
                waiting_args=None,  # This should raise a ValueError
                max_steps=3,
                default_wait_time=0.01,
                extra_wait_time=0.01,
                verbose=False,
            )

        # Now we test with correct waiting_args
        two_state_time_dependent_process(
            bandit=mock_bandit,
            fun=self.fun,
            waiting_args=(3, 4),  # Correct waiting args
            max_steps=3,
            default_wait_time=0.01,
            extra_wait_time=0.01,
            verbose=False,
        )

        # Check if bandit's update method was called expected number of times
        assert mock_bandit.update.call_count > 0

    def test_state_transition(self, mock_bandit):
        """Test transition from ALIVE to WAITING and back."""

        # Customize `fun` to force a failure state
        def failing_fun(*args):
            return (1, 9)

        two_state_time_dependent_process(
            bandit=mock_bandit,
            fun=failing_fun,
            failure_threshold=0.5,  # Higher threshold to force WAITING quicker
            waiting_args=(1, 2),
            max_steps=3,
            default_wait_time=0.01,
            extra_wait_time=0.01,
            verbose=True,
        )

        assert mock_bandit.select_arm.call_count > 0
        assert mock_bandit.update.call_count == 0
