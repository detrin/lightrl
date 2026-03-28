import random

from lightrl import EpsilonDecreasingBandit


def simulate_throughput(num_workers):
    optimal = 8
    efficiency = max(0, 1 - abs(num_workers - optimal) / optimal)
    noise = random.gauss(0, 0.05)
    return max(0, efficiency + noise)


if __name__ == "__main__":
    worker_counts = [1, 2, 4, 8, 12, 16, 24, 32]
    bandit = EpsilonDecreasingBandit(
        arms=worker_counts,
        initial_epsilon=1.0,
        limit_epsilon=0.05,
        half_decay_steps=50,
    )

    for step in range(500):
        arm = bandit.select_arm()
        reward = simulate_throughput(bandit.arms[arm])
        bandit.update(arm, reward)

    bandit.report()
    best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nOptimal worker count: {best}")
