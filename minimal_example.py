import random

from lightrl import EpsilonDecreasingBandit, two_state_time_dependent_process


def testing_simulation_function(num_tasks):
    p = min(1.0, max(0.0, 0.05 + num_tasks / 200 + random.uniform(-0.04, 0.04)))
    failed = num_tasks * p
    return num_tasks - failed, failed


if __name__ == "__main__":
    request_nums = list(range(10, 210, 10))
    bandit = EpsilonDecreasingBandit(
        arms=request_nums,
        initial_epsilon=1.0,
        limit_epsilon=0.1,
        half_decay_steps=len(request_nums) * 5,
    )
    print(bandit)
    two_state_time_dependent_process(
        bandit=bandit,
        fun=testing_simulation_function,
        failure_threshold=0.1,
        default_wait_time=0.1,
        extra_wait_time=0.1,
        waiting_args=[10],
        max_steps=1000,
        verbose=True,
        reward_factor=1e-6,
    )
