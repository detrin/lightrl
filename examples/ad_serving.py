import random

from lightrl import EpsilonFirstBandit

ADS = {
    "banner_top": 0.012,
    "sidebar_native": 0.025,
    "inline_content": 0.018,
    "footer_sticky": 0.008,
    "popup_exit_intent": 0.032,
}


def simulate_click(ad_idx):
    ctr = list(ADS.values())[ad_idx]
    return 1.0 if random.random() < ctr else 0.0


if __name__ == "__main__":
    bandit = EpsilonFirstBandit(
        arms=list(ADS.keys()),
        exploration_steps=200,
        epsilon=0.1,
    )

    for step in range(2000):
        arm = bandit.select_arm()
        reward = simulate_click(arm)
        bandit.update(arm, reward)

        if (step + 1) % 500 == 0:
            print(f"\n--- Step {step + 1} ---")
            bandit.report()

    best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nBest ad placement: {best}")
