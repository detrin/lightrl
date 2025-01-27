# Quickstart

## Installation

### Using PyPI

Using bare pip manager just simply install `lightrl` package.
```
pip install lightrl
```

## Minimal example
Imagine we have a time dependent process where we are processing tasks one by one (or in parallel) and that after too many requests our tasks will start to fail. Let's imagine we want to get below desired error rate of 10%. We also have `testing_simulation_function(num_tasks)` that will return the number of `successful_tasks` and number of `failed_tasks`. We can use `EpsilonDecreasingBandit` to optimize online the number of tasks per second. The function `two_state_time_dependent_process()` takes the bandit and keeps two states. The `ALIVE` state and `WAITING` state, the bandit is switching between those two states in order to probe the rewards (tasks per seconds multiplied by `reward_factor`). In `WAITING` state we can select lower number of tasks to process (`waiting_args`).
``` python
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
```
This will run for about 3 min and we will get the ouput
```
Q-values per arm:
  num_tasks=10: avg_reward=0.00006, count=9
  num_tasks=20: avg_reward=0.00004, count=5
  num_tasks=30: avg_reward=0.00003, count=3
  num_tasks=40: avg_reward=0.00008, count=10
  num_tasks=50: avg_reward=0.00009, count=7
  num_tasks=60: avg_reward=0.00010, count=8
  num_tasks=70: avg_reward=0.00011, count=38
  num_tasks=80: avg_reward=0.00011, count=6
  num_tasks=90: avg_reward=0.00010, count=22
  num_tasks=100: avg_reward=0.00011, count=160
  num_tasks=110: avg_reward=0.00009, count=7
  num_tasks=120: avg_reward=0.00009, count=9
  num_tasks=130: avg_reward=0.00010, count=8
  num_tasks=140: avg_reward=0.00010, count=6
  num_tasks=150: avg_reward=0.00008, count=8
  num_tasks=160: avg_reward=0.00005, count=13
  num_tasks=170: avg_reward=0.00004, count=6
  num_tasks=180: avg_reward=0.00002, count=3
  num_tasks=190: avg_reward=0.00000, count=3
  num_tasks=200: avg_reward=0.00000, count=8
```