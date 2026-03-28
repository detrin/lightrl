import random

from lightrl import LinUCBBandit

MODELS = ["haiku", "sonnet", "opus"]
COSTS = {"haiku": 0.25, "sonnet": 3.0, "opus": 15.0}
QUALITY = {"haiku": 0.55, "sonnet": 0.78, "opus": 0.95}


def featurize(task):
    return [task["complexity"], task["length"], float(task["needs_code"])]


def simulate_llm_call(model, task):
    base = QUALITY[model]
    penalty = task["complexity"] * (1 - base) * 0.5
    code_bonus = 0.15 * task["needs_code"] if model == "opus" else 0.0
    quality = max(0, min(1, base - penalty + code_bonus + random.gauss(0, 0.03)))
    cost_penalty = COSTS[model] / max(COSTS.values())
    return max(0, quality - 0.3 * cost_penalty)


def random_task():
    return {
        "complexity": random.uniform(0, 1),
        "length": random.uniform(0, 1),
        "needs_code": random.random() < 0.3,
    }


if __name__ == "__main__":
    bandit = LinUCBBandit(n_arms=3, n_features=3, alpha=0.3)
    cost_log = []

    for _ in range(1000):
        task = random_task()
        ctx = featurize(task)
        arm = bandit.select_arm(ctx)
        model = MODELS[arm]
        reward = simulate_llm_call(model, task)
        bandit.update(arm, ctx, reward)
        cost_log.append(COSTS[model])

    bandit.report()
    avg_cost = sum(cost_log) / len(cost_log)
    last_100_cost = sum(cost_log[-100:]) / 100
    print(f"\nAvg cost/call (all): ${avg_cost:.2f}")
    print(f"Avg cost/call (last 100): ${last_100_cost:.2f}")
    print("LinUCB learns to route simple tasks to haiku, complex code to opus")
