import time
from typing import Tuple

from lightrl import EpsilonDecreasingBandit, two_state_time_dependent_process


class SimulatedAPI:
    def __init__(self):
        # Initialize variables to keep track of requests
        self.time_window_requests = []
        self.window_length = 1  # 1 second window
        self.request_limit = 200  # request limit in a window
        self.block_duration = 1  # 1 second long block
        self.blocked_until = 0

    def request(self) -> Tuple[int, int]:
        current_time = time.time()

        # Remove requests older than the current window
        while (
            self.time_window_requests
            and self.time_window_requests[0] < current_time - self.window_length
        ):
            self.time_window_requests.pop(0)

        # Analyze if blocked
        if current_time < self.blocked_until:
            return 500

        if len(self.time_window_requests) > self.request_limit:  # Over the limit
            self.blocked_until = current_time + self.block_duration
            return 500

        self.time_window_requests.append(current_time)
        return 200

def api_request_fun(request_num):
    success_cnt = 0
    fail_cnt = 0
    for _ in range(request_num):
        http_status = api.request()
        if http_status == 200:
            success_cnt += 1
        else:
            fail_cnt += 1
        time.sleep(0.0001)
    return success_cnt, fail_cnt
    
if __name__ == "__main__":
    api = SimulatedAPI()
    request_nums = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    bandit = EpsilonDecreasingBandit(
        arms=request_nums, initial_epsilon=1.0, limit_epsilon=0.1, half_decay_steps=100
    )

    two_state_time_dependent_process(
        bandit=bandit,
        fun=api_request_fun,
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
