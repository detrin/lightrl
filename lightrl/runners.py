import time

from tqdm import tqdm


def _ensure_tuple(val):
    return val if isinstance(val, (tuple, list)) else (val,)


def two_state_time_dependent_process(
    bandit,
    fun,
    failure_threshold=0.1,
    default_wait_time=5,
    extra_wait_time=10,
    waiting_args=None,
    max_steps=500,
    verbose=False,
    reward_factor=1e-6,
):
    if waiting_args is None:
        raise ValueError("waiting_args must be provided")
    waiting_args = _ensure_tuple(waiting_args)

    state = "ALIVE"
    last_alive_successes = 0.0
    last_arm_index = None
    waiting_time = 0.0

    iterator = tqdm(range(max_steps)) if verbose else range(max_steps)

    for _ in iterator:
        if verbose:
            bandit.report()

        if state == "ALIVE":
            arm_idx = bandit.select_arm()
            fun_args = _ensure_tuple(bandit.arms[arm_idx])
            ok, fail = fun(*fun_args)
            time.sleep(default_wait_time)
            waiting_time += default_wait_time

            if fail / (ok + fail) >= failure_threshold:
                last_alive_successes = ok
                last_arm_index = arm_idx
                state = "WAITING"
            else:
                bandit.update(arm_idx, ok / waiting_time * reward_factor)
                waiting_time = 0.0
        else:
            ok, fail = fun(*waiting_args)
            time.sleep(default_wait_time + extra_wait_time)
            waiting_time += default_wait_time + extra_wait_time

            if fail / (ok + fail) < failure_threshold:
                bandit.update(last_arm_index, last_alive_successes / waiting_time * reward_factor)
                waiting_time = 0.0
                state = "ALIVE"

    if verbose:
        bandit.report()
