import random

from lightrl import EpsilonGreedyBandit

ARMS = ["endpoint_a", "endpoint_b", "endpoint_c"]


def reward_at_step(arm_idx, step):
    if step < 500:
        rates = [0.9, 0.3, 0.5]
    else:
        rates = [0.2, 0.8, 0.5]
    return max(0, rates[arm_idx] + random.gauss(0, 0.05))


if __name__ == "__main__":
    static = EpsilonGreedyBandit(arms=ARMS, epsilon=0.15)
    adaptive = EpsilonGreedyBandit(arms=ARMS, epsilon=0.15, ema_alpha=0.15)

    for step in range(1000):
        for arm_idx in range(3):
            r = reward_at_step(arm_idx, step)
            static.update(arm_idx, r)
            adaptive.update(arm_idx, r)

    print("Static (cumulative average) — stuck on pre-shift leader:")
    static.report()
    print(f"Best: {static.arms[static.q_values.index(max(static.q_values))]}")

    print("\nAdaptive (EMA alpha=0.15) — tracks the environment shift:")
    adaptive.report()
    print(f"Best: {adaptive.arms[adaptive.q_values.index(max(adaptive.q_values))]}")

    print("\nTrue best after step 500: endpoint_b")
