import random

from lightrl import BanditRouter, ThompsonBandit


def simulate_api_call(wait_seconds):
    recovery_prob = 1 - 0.9 * (0.5 ** (wait_seconds / 2))
    return 1.0 if random.random() < recovery_prob else 0.0


if __name__ == "__main__":
    wait_times = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    router = BanditRouter()
    router.register("backoff", ThompsonBandit(arms=wait_times))

    for _ in range(500):
        idx = router.select("backoff")
        wait = wait_times[idx]
        success = simulate_api_call(wait)
        effective_reward = success / (1 + wait / 10)
        router.update("backoff", idx, max(0, min(1, effective_reward)))

    router.report("backoff")
    bandit = router._bandits["backoff"]
    best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nOptimal backoff: {best}s")
