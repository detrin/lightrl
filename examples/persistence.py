import os
import random
import tempfile

from lightrl import ThompsonBandit


def simulate_click(ad_idx, rates):
    return 1.0 if random.random() < rates[ad_idx] else 0.0


if __name__ == "__main__":
    path = os.path.join(tempfile.gettempdir(), "bandit_state.json")
    ads = ["banner", "sidebar", "popup"]
    rates = [0.02, 0.05, 0.03]

    if os.path.exists(path):
        bandit = ThompsonBandit.load(path)
        print(f"Resumed from {path} (total pulls: {sum(bandit.counts)})")
    else:
        bandit = ThompsonBandit(arms=ads)
        print("Starting fresh")

    for _ in range(200):
        arm = bandit.select_arm()
        bandit.update(arm, simulate_click(arm, rates))

    bandit.save(path)
    print(f"Saved to {path} (total pulls: {sum(bandit.counts)})")
    bandit.report()
    print("\nRun again to see it resume from saved state")
