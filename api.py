import time

from lightrl import EpsilonDecreasingBandit, two_state_time_dependent_process


class SimulatedAPI:
    def __init__(self):
        self.time_window_requests: list[float] = []
        self.window_length = 1
        self.request_limit = 200
        self.block_duration = 1
        self.blocked_until = 0.0

    def request(self) -> int:
        now = time.time()
        while self.time_window_requests and self.time_window_requests[0] < now - self.window_length:
            self.time_window_requests.pop(0)
        if now < self.blocked_until:
            return 500
        if len(self.time_window_requests) > self.request_limit:
            self.blocked_until = now + self.block_duration
            return 500
        self.time_window_requests.append(now)
        return 200


def api_request_fun(request_num):
    ok, fail = 0, 0
    for _ in range(request_num):
        if api.request() == 200:
            ok += 1
        else:
            fail += 1
        time.sleep(0.0001)
    return ok, fail


if __name__ == "__main__":
    api = SimulatedAPI()
    bandit = EpsilonDecreasingBandit(
        arms=[10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        initial_epsilon=1.0,
        limit_epsilon=0.1,
        half_decay_steps=100,
    )
    two_state_time_dependent_process(
        bandit=bandit,
        fun=api_request_fun,
        failure_threshold=0.1,
        default_wait_time=0.1,
        extra_wait_time=0.1,
        waiting_args=[10],
        max_steps=1000,
        verbose=True,
        reward_factor=1e-6,
    )
