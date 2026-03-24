Evaluator Progression for the Knowledge QA Agent
The framework gives us two axes to think along:

Grader type: code-based → LLM-as-judge → human
Evaluation scope: item-level (output only) → trace-level (execution evidence)
Tier 1 — Code-based, Item-level (no LLM, no trace)
These are the cheapest, fastest, and most deterministic. Run them on every experiment.

1. Non-empty response check
The simplest possible gate: did the agent return anything at all? Catches crashes, timeouts, and "Error: ..." strings.

# score 1.0 if output is non-empty and doesn't start with "Error:"
2. Substring match
Does the expected answer literally appear in the response? Works well for exact-value questions (e.g., "2008", "Federal Reserve"). Zero cost, instant.

# score 1.0 if expected_output.lower() in output.lower()
Limitation: will miss paraphrased correct answers — that's why you need Tier 3.

3. Source citation presence
Does the output include at least one URL or citation marker? The agent's GroundedResponse.sources is rich metadata, but you can also check the raw output string. A response with zero sources is more likely to be hallucinated.

# score 1.0 if "http" appears in output, or len(response.sources) > 0
Tier 2 — Code-based, Trace-level (no LLM, uses extract_trace_metrics)
These require the two-pass run_experiment_with_trace_evals pattern from notebook 02, but still no LLM call — just arithmetic on the trace object.

4. Latency evaluator
How long did the agent take? Already demonstrated in 02_evaluation_harness.ipynb:

metrics = extract_trace_metrics(trace)
return [Evaluation(name="latency_sec", value=metrics.latency_sec)]
Useful for flagging when replanning or long context compaction blows up wall time.

5. Token usage / cost evaluator
Total input + output tokens → estimated cost. TraceMetrics has .tokens and .cost. Tracks the cost-performance dimension directly. Lets you compare gemini-2.5-flash vs gemini-2.5-pro experiments on a $/F1 basis.

6. Tool call count evaluator
metrics.tool_call_count — a proxy for research depth. An agent that uses 0 tools answered from parametric memory (risky for a grounded QA agent). An agent that uses 30+ may be inefficient. A healthy band exists between them.

7. Tool type diversity evaluator
Did the agent use only google_search, or did it also call web_fetch / read_file? The PlanReAct agent is designed to go deep on sources — if it never fetches pages, it's skimming.

tool_names_used = {obs.name for obs in trace.observations if obs.type == "SPAN"}
used_fetch = "web_fetch" in tool_names_used
Tier 3 — LLM-as-judge, Item-level (output only, no trace)
These require an LLM call per item. The key distinction from Tier 1: they handle semantic equivalence that string matching cannot.

8. DeepSearchQA grader (already provided)
The main outcome metric. F1 / precision / recall. Handles both Single Answer and Set Answer types. The LLM checks semantic containment, not exact string match. This is your primary quality signal.

9. Answer conciseness rubric
Custom rubric via create_llm_as_judge_evaluator. Some categories (Finance, Science) expect precise answers; others allow narrative.

- **conciseness**: 1 if the answer gives the core facts without unnecessary padding,
  0 if the response is longer than ~3 paragraphs for a factual question
- **no_hedging**: 1 if the agent commits to an answer rather than saying
  "I could not find...", 0 otherwise
This catches the correct_with_extraneous failure mode at the prose level.

10. Per-category correctness breakdown
Not a new evaluator — but a different use of the DeepSearchQA grader. Pass metadata.category into your results table. The DeepSearchQA dataset has 17 categories (Finance & Economics, History, Science, etc.). You may find the agent is strong on Finance but weak on Science — which maps to a concrete prompt or tooling improvement.

Tier 4 — LLM-as-judge, Trace-level (uses trace evidence)
Requires two-pass evaluation. The LLM judge has access to what the agent actually did, not just what it said.

11. Trace groundedness (already provided)
create_trace_groundedness_evaluator. Breaks the answer into atomic claims and checks each against tool observations from the trace. Score = (supported claims) / (total claims). This is the hallucination detector for the grounded QA use case — an answer can score well on F1 but still be fabricated if the right answer happened to be in the model's training data.

12. Research plan quality
The KnowledgeGroundedAgent with enable_planning=True emits a numbered research plan before tool execution. You can extract it from the trace (it's in span metadata or model thought events). A custom LLM-as-judge rubric then assesses:

- **plan_specificity**: 1 if each plan step targets a specific sub-question, 0 if steps are vague
- **plan_coverage**: 1 if the plan addresses all parts of a multi-part question, 0 if it misses a component
- **plan_efficiency**: 1 if the plan avoids redundant steps upfront, 0 if multiple steps are essentially duplicates
This is valuable because a bad plan causes downstream failure even if each tool call is correct.

13. Tool call redundancy evaluator
Did the agent issue near-duplicate search queries? e.g., "Federal Reserve interest rate 2023" followed by "Fed rate hike 2023". Extract all google_search inputs from trace observations, then ask an LLM judge:

Given this list of search queries, score redundancy:
- 0.0 = all queries are distinct and complementary
- 1.0 = several queries are semantically equivalent (wasted API calls)
This operationalizes the Tool Usage Quality dimension directly.

Tier 5 — Composite / Most Complex
These combine multiple signals or require custom orchestration.

14. Plan-execution alignment
Compare the declared research plan (from the agent's opening response) against the actual tool calls in the trace. An LLM judge answers: did the agent follow its own plan, or did it deviate, skip steps, or replan mid-execution? Requires extracting both the plan text and the ordered list of tool calls from the trace.

15. Efficiency score: quality / cost
A derived scalar: F1 / (cost_usd + epsilon). No LLM call needed — just combine outputs from the DeepSearchQA grader (Tier 3) and the cost evaluator (Tier 2). Lets you plot a Pareto frontier across model configurations and answer the question: is gemini-2.5-pro worth the extra cost over flash for this benchmark?

Summary Table
#	Evaluator	Type	Scope	Grader	What it measures
1	Non-empty check	Code	Item	—	Outcome: basic liveness
2	Substring match	Code	Item	—	Outcome: exact recall
3	Source citation presence	Code	Item	—	Tool usage: grounding signal
4	Latency	Code	Trace	—	Cost-performance
5	Token / cost	Code	Trace	—	Cost-performance
6	Tool call count	Code	Trace	—	Tool usage: depth
7	Tool type diversity	Code	Trace	—	Tool usage: breadth
8	DeepSearchQA grader (provided)	LLM	Item	LLM-judge	Outcome: F1/P/R
9	Conciseness rubric	LLM	Item	LLM-judge	Reasoning: verbosity
10	Per-category breakdown	LLM	Item	(reuses #8)	Outcome: gap analysis
11	Trace groundedness (provided)	LLM	Trace	LLM-judge	Outcome: hallucination
12	Research plan quality	LLM	Trace	LLM-judge	Reasoning: plan coherence
13	Tool call redundancy	LLM	Trace	LLM-judge	Tool usage: efficiency
14	Plan-execution alignment	LLM	Trace	LLM-judge	Reasoning: follow-through
15	Efficiency score (F1/cost)	Code	Item+Trace	—	Cost-performance composite
The two already-implemented evaluators (#8 and #11) cover the most important dimensions. The highest-leverage new evaluator to build next would be #12 (research plan quality) — because plan failures cascade into all downstream metrics, and it's not observable from the output alone.