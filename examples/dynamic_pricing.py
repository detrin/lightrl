import random

from lightrl import ThompsonBandit

PRICES = [9.99, 14.99, 19.99, 24.99, 29.99]
BASE_CONVERSIONS = [0.35, 0.28, 0.20, 0.12, 0.06]


def simulate_purchase(price_idx, step):
    season_boost = 0.1 * (1 + __import__("math").sin(2 * 3.14159 * step / 200))
    conv_rate = min(1.0, BASE_CONVERSIONS[price_idx] + season_boost)
    purchased = random.random() < conv_rate
    revenue = PRICES[price_idx] if purchased else 0.0
    return min(1.0, revenue / max(PRICES))


if __name__ == "__main__":
    bandit = ThompsonBandit(arms=PRICES)
    total_revenue = 0.0

    for step in range(2000):
        arm = bandit.select_arm()
        reward = simulate_purchase(arm, step)
        bandit.update(arm, reward)
        total_revenue += reward * max(PRICES)

    bandit.report()
    best = PRICES[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nBest price point: ${best}")
    print(f"Total normalized revenue: {total_revenue:.0f}")
    print("Thompson adapts to seasonal conversion shifts without manual repricing")
