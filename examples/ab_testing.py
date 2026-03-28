import random

from lightrl import EpsilonGreedyBandit

VARIANTS = {
    "control": 0.02,
    "red_button": 0.035,
    "large_cta": 0.05,
    "minimal_layout": 0.04,
}


def simulate_conversion(variant_idx):
    rate = list(VARIANTS.values())[variant_idx]
    return 1.0 if random.random() < rate else 0.0


if __name__ == "__main__":
    bandit = EpsilonGreedyBandit(arms=list(VARIANTS.keys()), epsilon=0.15)

    for step in range(5000):
        arm = bandit.select_arm()
        reward = simulate_conversion(arm)
        bandit.update(arm, reward)

        if (step + 1) % 1000 == 0:
            print(f"\n--- Step {step + 1} ---")
            bandit.report()

    best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nBest variant: {best}")
