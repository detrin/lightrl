import random

from lightrl import UCB1Bandit


def simulate_model_accuracy(lr):
    optimal_lr = 0.001
    score = max(0, 1 - 50 * abs(lr - optimal_lr))
    return min(1.0, max(0.0, score + random.gauss(0, 0.02)))


if __name__ == "__main__":
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    bandit = UCB1Bandit(arms=learning_rates)

    for step in range(200):
        arm = bandit.select_arm()
        reward = simulate_model_accuracy(bandit.arms[arm])
        bandit.update(arm, reward)

    bandit.report()
    best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nBest learning rate: {best}")
