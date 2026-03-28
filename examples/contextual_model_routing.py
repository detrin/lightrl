import random

from lightrl import LinUCBBandit

MODELS = ["haiku", "sonnet", "opus"]


def featurize(task_complexity, input_length, is_code):
    return [task_complexity, input_length, 1.0 if is_code else 0.0]


def simulate_quality(model_idx, features):
    complexity, length, is_code = features
    base = {0: 0.4, 1: 0.7, 2: 0.95}[model_idx]
    cost = {0: 0.0, 1: 0.15, 2: 0.4}[model_idx]
    complexity_penalty = complexity * (1 - base)
    code_bonus = 0.2 * is_code if model_idx == 2 else 0.0
    score = base - complexity_penalty + code_bonus - cost
    return max(0, min(1, score + random.gauss(0, 0.05)))


if __name__ == "__main__":
    bandit = LinUCBBandit(n_arms=3, n_features=3, alpha=0.5)

    tasks = []
    for _ in range(500):
        complexity = random.uniform(0, 1)
        length = random.uniform(0, 1)
        is_code = random.random() < 0.4
        ctx = featurize(complexity, length, is_code)
        arm = bandit.select_arm(ctx)
        reward = simulate_quality(arm, ctx)
        bandit.update(arm, ctx, reward)
        tasks.append((complexity, is_code, MODELS[arm]))

    bandit.report()

    simple_text = [t[2] for t in tasks[-100:] if t[0] < 0.3 and not t[1]]
    complex_code = [t[2] for t in tasks[-100:] if t[0] > 0.7 and t[1]]

    if simple_text:
        best = max(set(simple_text), key=simple_text.count)
        print(f"\nSimple text tasks -> mostly routed to: {best}")
    if complex_code:
        best = max(set(complex_code), key=complex_code.count)
        print(f"Complex code tasks -> mostly routed to: {best}")
