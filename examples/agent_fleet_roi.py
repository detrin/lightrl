import random

from lightrl import BanditRouter, ThompsonBandit

AGENTS = ["researcher", "coder", "reviewer", "planner"]
BASE_SUCCESS = {"researcher": 0.7, "coder": 0.85, "reviewer": 0.6, "planner": 0.5}
AGENT_COST = {"researcher": 0.8, "coder": 1.2, "reviewer": 0.5, "planner": 0.3}


def simulate_agent_task(agent, task_type):
    base = BASE_SUCCESS[agent]
    match_bonus = (
        0.2
        if (
            (agent == "coder" and task_type == "implementation")
            or (agent == "researcher" and task_type == "analysis")
            or (agent == "reviewer" and task_type == "qa")
            or (agent == "planner" and task_type == "design")
        )
        else -0.1
    )
    quality = max(0, min(1, base + match_bonus + random.gauss(0, 0.05)))
    cost_ratio = AGENT_COST[agent] / max(AGENT_COST.values())
    return max(0, quality - 0.2 * cost_ratio)


TASK_TYPES = ["implementation", "analysis", "qa", "design"]

if __name__ == "__main__":
    router = BanditRouter()
    for tt in TASK_TYPES:
        router.register(tt, ThompsonBandit(arms=AGENTS))

    task_log: dict[str, list[str]] = {tt: [] for tt in TASK_TYPES}

    for _ in range(500):
        tt = random.choice(TASK_TYPES)
        arm = router.select(tt)
        agent = AGENTS[arm]
        reward = simulate_agent_task(agent, tt)
        router.update(tt, arm, reward)
        task_log[tt].append(agent)

    router.report()

    print("\nLearned routing (last 50 tasks per type):")
    for tt in TASK_TYPES:
        recent = task_log[tt][-50:]
        if recent:
            best = max(set(recent), key=recent.count)
            pct = recent.count(best) / len(recent) * 100
            print(f"  {tt} -> {best} ({pct:.0f}% of recent)")
