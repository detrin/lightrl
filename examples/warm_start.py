import random

from lightrl import EpsilonGreedyBandit

TRUE_RATES = [0.02, 0.05, 0.08, 0.04]
ENDPOINTS = ["us-east", "eu-west", "ap-south", "us-west"]


def simulate_success(idx):
    return 1.0 if random.random() < TRUE_RATES[idx] else 0.0


if __name__ == "__main__":
    naive = EpsilonGreedyBandit(arms=ENDPOINTS, epsilon=0.1)
    informed = EpsilonGreedyBandit(arms=ENDPOINTS, epsilon=0.1, priors=[0.01, 0.04, 0.07, 0.03])

    for _ in range(200):
        for b in [naive, informed]:
            arm = b.select_arm()
            b.update(arm, simulate_success(arm))

    print("Naive (no priors):")
    naive.report()
    best_naive = naive.arms[naive.q_values.index(max(naive.q_values))]

    print("\nInformed (with priors):")
    informed.report()
    best_informed = informed.arms[informed.q_values.index(max(informed.q_values))]

    print(f"\nNaive picked: {best_naive}")
    print(f"Informed picked: {best_informed}")
    print(f"True best: {ENDPOINTS[TRUE_RATES.index(max(TRUE_RATES))]}")
