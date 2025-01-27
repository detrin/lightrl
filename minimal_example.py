import time
import random

from lightrl import EpsilonDecreasingBandit, two_state_time_dependent_process


def testing_simulation_function(num_tasks):
    # Simulate the number of successful and failed tasks
    # num_tasks = 0, p = 0.05
    # num_tasks = 100, p = 0.05 + 100 / 200 = 0.55
    # num_tasks = 200, p = 0.05 + 200 / 200 = 1.05
    p = 0.05 + num_tasks / 200
    noise = random.uniform(-0.04, 0.04)
    p_with_noise = p + noise
    p_with_noise = min(1.0, max(0.0, p_with_noise))
    failed_tasks = num_tasks * p_with_noise
    successful_tasks = num_tasks - failed_tasks
    return successful_tasks, failed_tasks

if __name__ == "__main__":
    request_nums = list(range(10, 210, 10))
    bandit = EpsilonDecreasingBandit(
        arms=request_nums, initial_epsilon=1.0, limit_epsilon=0.1, half_decay_steps=len(request_nums) * 5
    )
    print(bandit)

    two_state_time_dependent_process(
        bandit=bandit,
        fun=testing_simulation_function,
        failure_threshold=0.1,  # Allowed failure is 10%
        default_wait_time=0.1,  # Wait 0.1 s between requests
        extra_wait_time=0.1,  # Wait extra 0.1 s when in blocked state
        waiting_args=[
            10
        ],  # Working with only 10 requests in the waiting state to test if we are still blocked
        max_steps=1000,  # Run for maximum of 1000 steps
        verbose=True,
        reward_factor=1e-6,  # In case you want to keep reward below 1 (for UCB1Bandit)
    )