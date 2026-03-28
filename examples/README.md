# Examples

## Classics

### minimal_example.py
Two-state time-dependent process with a simulated failure rate that scales with batch size.
EpsilonDecreasingBandit learns the optimal batch size while balancing exploration/exploitation
over 1000 steps.

### ab_testing.py
A/B test across 4 landing page variants with different conversion rates.
EpsilonGreedyBandit allocates traffic toward the winning variant while maintaining
a 15% exploration rate to detect shifts.

### ad_serving.py
Ad placement optimization across 5 positions with different click-through rates.
EpsilonFirstBandit explores uniformly for 200 steps, then switches to epsilon-greedy
exploitation to maximize CTR.

### resource_allocation.py
Dynamic worker pool sizing for a compute cluster. EpsilonDecreasingBandit starts with
heavy exploration and gradually converges on the optimal worker count as epsilon decays.

### hyperparameter_search.py
Learning rate selection via bandit-based search. UCB1Bandit balances exploration of
untested rates with exploitation of promising ones using upper confidence bounds.
Rewards are normalized to [0, 1] as required by UCB1.

### network_routing.py
Endpoint selection across 4 regions with varying latency profiles.
GreedyBanditWithHistory maintains a sliding window of 30 observations per endpoint,
adapting to latency changes over time.

## Agent & LLM

### prompt_template_selection.py
BanditRouter manages separate bandits per task type (code_gen, summarize, qa), each
learning which prompt template produces the best output. Demonstrates how an LLM agent
can offload template selection to lightrl instead of burning tokens reasoning about it.

### agent_router.py
Full agent loop with BanditRouter managing two independent decisions (model selection
and batch size) using different bandit strategies. Shows save/load of the entire
router with mixed bandit types (ThompsonBandit + UCB1Bandit).

### contextual_model_routing.py
LinUCB contextual bandit routes tasks to LLM models based on feature vectors
(complexity, length, is_code). Learns different routing policies for different
task types — e.g., simple text to cheaper models, complex code to opus.

### llm_cost_optimizer.py
LinUCB contextual bandit routes LLM API calls to haiku/sonnet/opus based on
task features (complexity, length, needs_code). Optimizes the quality-vs-cost
tradeoff per request — sends simple tasks to cheap models, complex code to opus.

### agent_fleet_roi.py
BanditRouter with per-task-type ThompsonBandits learns which AI agent
(researcher, coder, reviewer, planner) to dispatch for each task category.
Converges on the optimal agent-task mapping while maximizing ROI.

## Infrastructure

### retry_backoff.py
Learns the optimal retry wait time for a flaky API using BanditRouter + ThompsonBandit.
Reward balances recovery probability against wait cost — shorter waits score higher
when they succeed. Converges on the sweet spot without hardcoding exponential backoff.

### db_query_routing.py
Per-table-size EpsilonGreedyBandits with EMA decay learn the fastest query
execution strategy (seq_scan, index_scan, hash_join, merge_join) for different
data volumes. Adapts if workload patterns shift over time.

### dynamic_pricing.py
ThompsonBandit learns optimal price points for an e-commerce product with
seasonal conversion shifts. Adapts pricing without manual repricing rules —
balances revenue per sale against conversion probability across 5 price tiers.

## Techniques

### warm_start.py
Side-by-side comparison of a naive bandit vs one initialized with prior beliefs.
The informed bandit starts exploiting the correct arm sooner because priors
encode domain knowledge (e.g., "ap-south is probably fastest").

### ema_nonstationary.py
Demonstrates EMA decay vs cumulative average when the environment changes at step 500.
Static averaging blurs pre- and post-shift rewards into an ambiguous tie, while
EMA (alpha=0.15) correctly tracks that the best arm switched.

### persistence.py
Save/load a ThompsonBandit to JSON. Run it once to train, run again to resume from
saved state. Demonstrates how bandits survive process restarts with zero-dependency
serialization.
