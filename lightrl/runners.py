import time
from typing import Callable, Optional, Tuple, Union, List
from tqdm import tqdm


def two_state_time_dependent_process(
    bandit,
    fun: Callable[..., Tuple[float, float]],
    failure_threshold: float = 0.1,
    default_wait_time: float = 5,
    extra_wait_time: float = 10,
    waiting_args: Optional[Union[Tuple, List]] = None,
    max_steps: int = 500,
    verbose: bool = False,
    reward_factor: float = 1e-6,
) -> None:
    """Execute a two-state time-dependent process with a bandit decision-maker.

    This function simulates a process which alternates between an "ALIVE" state
    and a "WAITING" state based on the performance of a given task in relation
    to a failure threshold. It updates the bandit model with rewards calculated
    from successful tasks.

    Args:
        bandit: An object with methods `select_arm`, `update`, and `report`, representing
                a multi-armed bandit.
        fun: A function to be called with the current arm's arguments. Should return a tuple
             containing the number of successful and failed tasks.
        failure_threshold: A float to determine what fraction of tasks fails that triggers
                           a switch to the "WAITING" state.
        default_wait_time: The base wait time in seconds between task executions in the "ALIVE" state.
        extra_wait_time: Additional wait time in seconds to be added in the "WAITING" state.
        waiting_args: Arguments to be used when calling `fun` in the "WAITING" state.
        max_steps: Maximum number of iterations/steps to be performed.
        verbose: If True, prints additional detailed logs and progress via tqdm.
        reward_factor: A scaling factor to adjust the magnitude of the reward computed.

    Raises:
        ValueError: If `waiting_args` is not provided or if it is not of expected types.
    """

    if waiting_args is None:
        raise ValueError("waiting_args must be provided")
    else:
        if not (isinstance(waiting_args, tuple) or isinstance(waiting_args, list)):
            waiting_args = (waiting_args,)

    state = "ALIVE"
    last_alive_successes: float = 0.0
    last_arm_index: Optional[int] = None
    waiting_steps: int = 0
    waiting_time: float = 0.0

    iterator = range(max_steps)
    if verbose:
        iterator = tqdm(range(max_steps))

    for _ in iterator:
        if verbose:
            bandit.report()

        if state == "ALIVE":
            current_arm_index = bandit.select_arm()

            fun_args = bandit.arms[current_arm_index]
            if not (isinstance(fun_args, tuple) or isinstance(fun_args, list)):
                fun_args = (fun_args,)
            successful_tasks, failed_tasks = fun(*fun_args)
            fail_fraction = failed_tasks / (successful_tasks + failed_tasks)

            time.sleep(default_wait_time)
            waiting_time += default_wait_time

            if fail_fraction >= failure_threshold:
                last_alive_successes = successful_tasks
                last_arm_index = current_arm_index
                state = "WAITING"
                waiting_steps = 0
            else:
                reward = successful_tasks / waiting_time * reward_factor
                bandit.update(current_arm_index, reward)
                waiting_time = 0.0

        else:
            successful_tasks, failed_tasks = fun(*waiting_args)
            fail_fraction = failed_tasks / (successful_tasks + failed_tasks)
            waiting_steps += 1

            time.sleep(default_wait_time + extra_wait_time)
            waiting_time += default_wait_time + extra_wait_time

            if fail_fraction < failure_threshold:
                reward = last_alive_successes / waiting_time * reward_factor
                bandit.update(last_arm_index, reward)
                waiting_time = 0.0
                state = "ALIVE"

    if verbose:
        bandit.report()
