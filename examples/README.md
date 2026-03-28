# Examples

## minimal_example.py
Two-state time-dependent process with a simulated failure rate that scales with batch size.
EpsilonDecreasingBandit learns the optimal batch size while balancing exploration/exploitation
over 1000 steps.

## ab_testing.py
A/B test across 4 landing page variants with different conversion rates.
EpsilonGreedyBandit allocates traffic toward the winning variant while maintaining
a 15% exploration rate to detect shifts.

## ad_serving.py
Ad placement optimization across 5 positions with different click-through rates.
EpsilonFirstBandit explores uniformly for 200 steps, then switches to epsilon-greedy
exploitation to maximize CTR.

## resource_allocation.py
Dynamic worker pool sizing for a compute cluster. EpsilonDecreasingBandit starts with
heavy exploration and gradually converges on the optimal worker count as epsilon decays.

## hyperparameter_search.py
Learning rate selection via bandit-based search. UCB1Bandit balances exploration of
untested rates with exploitation of promising ones using upper confidence bounds.
Rewards are normalized to [0, 1] as required by UCB1.

## network_routing.py
Endpoint selection across 4 regions with varying latency profiles.
GreedyBanditWithHistory maintains a sliding window of 30 observations per endpoint,
adapting to latency changes over time.

## retry_backoff.py
Learns the optimal retry wait time for a flaky API using BanditRouter + ThompsonBandit.
Reward balances recovery probability against wait cost — shorter waits score higher
when they succeed. Converges on the sweet spot without hardcoding exponential backoff.

## prompt_template_selection.py
Agent-facing example: BanditRouter manages separate bandits per task type (code_gen,
summarize, qa), each learning which prompt template produces the best output.
Demonstrates how an LLM agent can offload template selection to lightrl instead of
burning tokens reasoning about it.
