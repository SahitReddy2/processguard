# Judge audit — LLM-based detectors

The two LLM-judge detectors in ProcessGuard (`ReasoningActionMismatchDetector`
and `PrematureTerminationDetector`) wrap a small Anthropic call that returns
a yes/no verdict on whether a coordination failure has occurred. This
document audits the exact prompts shipped in v0.1.0, enumerates the most
plausible ways each one will produce a wrong verdict, notes what test
coverage they currently have, and proposes a small evaluation harness to
convert these hypotheses into measurements.

The failure cases listed below are **plausible, not verified**. The
proposed harness exists specifically to learn which of these actually fire
and at what rate, and to give us a baseline we can defend if these prompts
change in future releases.

---

## ReasoningActionMismatchDetector (FM-2.6)

**File:** `processguard/detectors/reasoning_action_mismatch.py` (the `_judge`
method, lines 58–99).

### Exact prompt

The detector buffers `REASONING` events keyed by `(trace_id, agent_name)`.
When the next `TOOL_CALL` arrives for the same key, it sends the following
user message to `claude-haiku-4-5-20251001` with `max_tokens=80`:

```
Does this action match the stated reasoning?

REASONING: {reasoning[:500]}

ACTION: Called tool '{tool_name}' with args {tool_args}

Reply with exactly: MATCH or MISMATCH, then a confidence score 0-10.
```

The reasoning is clipped to 500 characters. The action string is built from
`f"Called tool '{tool_name}' with args {tool_args}"` and is **not** clipped,
so a `tool_args` dict with a large value (e.g. a long file path or a large
JSON blob) ends up in the prompt verbatim.

Parsing the response:
1. The reply is uppercased and tested for the substring `"MISMATCH"`. If
   absent, no detection fires.
2. The first integer in the reply is extracted via `re.search(r"(\d+)", text)`
   and divided by 10 to produce a confidence score. If no integer is found,
   confidence defaults to 0.7.
3. Detections below `confidence_floor` (default 0.5) are suppressed.

### Three plausible failure cases

**1. False positive — multi-step plan, first step executed correctly.**
Reasoning: *"First I'll search for the latest papers on X, then I'll read
the top three results, then I'll synthesize a summary."* Action:
`web_search(query="X recent papers")`. The action correctly executes step
one of the three-step plan, but the judge sees an action that covers only
a fraction of the stated reasoning and is likely to call this MISMATCH
because the reading and synthesizing steps are absent from the action.
This is the case named in the detector's contract limitation sentence,
and is also the most likely to occur in practice because multi-step
planning is a common pattern in well-prompted agents.

**2. False negative — same tool, semantically different intent.**
Reasoning: *"I need to find the latest research papers on X, not tutorials
or blog posts."* Action: `web_search(query="X")`. The action uses the right
tool but with a query that has no filter against tutorials or blogs. A
strict judge should flag this as a mismatch (the action does not encode
the "not tutorials" constraint), but Claude Haiku will probably call this
MATCH because the action is searching about X, which is broadly consistent
with the stated topic. The detector is most useful exactly when intent
*looks* aligned with action but isn't, and this is the case it is most
likely to miss.

**3. Truncation and action-blob artefacts.**
Reasoning is clipped to 500 characters. When the controlling intent lives
in the tail of a longer reasoning trace (*"…and finally, I should delegate
to the writer agent rather than searching myself"*), the judge compares
an action against a context where the controlling intent has been cut
off, and the verdict becomes effectively meaningless. Conversely, the
unclipped action string can swell to thousands of characters when a tool
takes a large argument, pushing the prompt structure off-balance — the
judge then sees a small REASONING block and a huge ACTION block, which
tends to bias verdicts toward MISMATCH because the asymmetry suggests
the agent is doing more than it said it would.

### Mocking status

**No tests exist for this detector.** There is no
`tests/test_reasoning_action_mismatch.py`, no mocked-LLM wiring test, and no
quality measurement. The v0.1.0 test count of 24 covers `step_repetition`,
`unaware_termination`, `no_progress_loop`, and `policy` only. This means:

- The detector's wiring (buffering REASONING events, firing on the next
  TOOL_CALL, honouring `confidence_floor`) is not verified by any
  automated test.
- The parsing logic (substring check for `"MISMATCH"`, the `\d+` regex for
  confidence, the default-0.7 fallback) is not verified.
- The judge's *quality* — the question this audit is concerned with — is
  also not measured.

### What a real evaluation would tell us

A 10-case evaluation set (five true positives, five true negatives) run
against the real model would tell us, concretely, what fraction of
multi-step-plan reasoning traces the judge incorrectly calls MISMATCH,
what fraction of same-tool semantic-drift cases it incorrectly calls
MATCH, and where the parsing fallback (default confidence 0.7) sits
relative to the judge's true confidence distribution — turning the three
hypotheses above into a small, defensible precision/recall pair we can
publish alongside the detector.

---

## PrematureTerminationDetector (FM-3.1)

**File:** `processguard/detectors/premature_termination.py` (the `_judge`
method, lines 72–127).

### Exact prompt

The detector tracks `MESSAGE` events per `(trace_id, agent_name)` to
maintain the agent's most recent output. When a `TERMINATE` event arrives,
it sends the following user message to `claude-haiku-4-5-20251001` with
`max_tokens=120`:

```
Has this output fully addressed the original task?

TASK: {task[:400]}

OUTPUT: {output[:400]}

Reply: COMPLETE or INCOMPLETE, then confidence 0-10, then one sentence why.
```

Both `task` and `output` are clipped to 400 characters.

Parsing is structurally identical to the RAM detector:
1. Substring check for `"INCOMPLETE"` in the uppercased reply.
2. First integer extracted as confidence (× 0.1), default 0.7 if none found.
3. Suppressed if confidence < `confidence_floor` (default 0.5).

There is also an explicit-bypass path (lines 73–85): if the agent's last
`MESSAGE` content is empty when TERMINATE arrives, the detector emits a
high-confidence (0.9) detection *without* calling the LLM. That path is
not exercised by the audit but should be on the test list whenever
fixtures get written.

### Three plausible failure cases

**1. False negative — confident-tone single paragraph for a multi-part task.**
Task: *"Find five recent papers on retrieval-augmented generation, summarize
each, and recommend the most relevant one for a customer-support chatbot."*
Output: *"Retrieval-augmented generation has emerged as a strong technique
for grounding LLM outputs in factual sources. Recent work has focused on
chunking strategies, retrieval quality, and re-ranking. For customer
support, a RAG system with good retrieval and conservative generation is
typically the right choice."* This is one confident paragraph that reads
like a finished answer. It contains no citations, no per-paper summaries,
and the recommendation is generic. Claude Haiku may rate this COMPLETE
because the prose has the *shape* of a conclusion. This is the case
named in the contract's limitation sentence.

**2. False positive — terse-but-correct output for a fact-shaped task.**
Task: *"What's the population of Tokyo?"* Output: *"13.96 million (2024
estimate)."* The output is genuinely complete: it answers the question
directly and cites the year. A judge biased toward verbose explanations
may rate this INCOMPLETE because the output does not explain methodology
or alternatives. This is the inverse of the contract's "must not fire"
example, and measurement would tell us whether the judge actually does
mis-call this case in practice.

**3. Truncation interaction.**
Both task and output are clipped to 400 characters. For long tasks (e.g.
multi-paragraph project briefs) or long outputs (e.g. multi-section
reports), the judge sees only the lead of each. A six-paragraph report
that fully addresses a three-paragraph task is then compared as
lead-of-report vs. lead-of-task: a judge that finds the intro paragraphs
broadly aligned may call this COMPLETE even when the body of the report
fails to deliver on the body of the task, and a judge that finds the
intros mismatched may call INCOMPLETE for the opposite reason. The
truncation creates a structurally different failure mode from the two
above and may be the single biggest source of judge error on realistic
agent traces, where tasks and outputs frequently exceed 400 characters.

### Mocking status

**No tests exist for this detector.** No
`tests/test_premature_termination.py`, no mocked-LLM wiring test, no
empty-output-bypass test, no quality measurement. Same situation as
ReasoningActionMismatch.

### What a real evaluation would tell us

A 10-case set (five true positives covering the three failure modes
above, five true negatives covering brief-but-complete answers and
well-shaped multi-part deliverables) run against the real model would
tell us whether the truncation-induced failure (case 3) dominates the
confident-tone failure (case 1), and where the `confidence_floor` of
0.5 falls on the judge's actual confidence distribution for these
inputs — both numbers that would inform whether the default policy on
FM-3.1 should be `LOG`, `STEER`, or `HALT`.

---

## Proposed evaluation harness

This is a **shape proposal only**. Fixtures are deliberately not being
written in this session — handcrafting 20 calibration cases that exercise
the failure modes above without being trivially gameable is a focused
1–2 hour task that deserves its own session.

### Layout

```
tests/fixtures/judges/
  reasoning_action_mismatch.jsonl    # 10 cases: 5 TP, 5 TN
  premature_termination.jsonl        # 10 cases: 5 TP, 5 TN

scripts/eval_judges.py               # CLI runner
eval_results/                        # gitignored; per-run reports
```

### Fixture schema

One JSON object per line.

For ReasoningActionMismatch:

```json
{
  "id": "ram-01",
  "reasoning": "First I'll search for the latest papers on X, then read the top three, then synthesize a summary.",
  "tool_name": "web_search",
  "tool_args": {"query": "X recent papers"},
  "expected": "match",
  "category": "multi-step-plan-first-step",
  "rationale_for_human": "Action correctly executes step 1 of 3. Judge will likely (wrongly) say MISMATCH."
}
```

For PrematureTermination:

```json
{
  "id": "pt-01",
  "task": "What's the population of Tokyo?",
  "output": "13.96 million (2024 estimate).",
  "expected": "complete",
  "category": "terse-correct-fact",
  "rationale_for_human": "Output is brief but fully addresses the task. Judge may (wrongly) call INCOMPLETE."
}
```

The `category` tag groups cases by the failure mode they exercise, so the
report can break out per-category precision/recall rather than only the
single aggregate number. This is what makes the eval informative for
debugging the prompt, not just for grading it.

### Script behaviour

```
python scripts/eval_judges.py --detector reasoning_action_mismatch
python scripts/eval_judges.py --detector premature_termination
python scripts/eval_judges.py --all
```

For each case:
1. Construct the `AgentEvent`(s) the detector expects (a REASONING + TOOL_CALL
   pair for RAM; a MESSAGE + TERMINATE pair plus `set_task` for PT).
2. Pass through the detector's real `_judge()` method. **No mocking** — the
   whole point of the harness is to measure the real model.
3. Compare the verdict against `expected`.
4. Record both the binary match and the judge's raw reply (for human
   inspection of failures).

Output:
- Per-detector confusion matrix.
- Precision, recall, F1 — aggregate and per-category.
- Per-case verdict log (id, expected, actual, confidence, judge raw text,
  any parsing fallback that triggered).
- Cost summary in dollars.
- Written to `eval_results/{detector}_{ISO-timestamp}.json` for tracking
  judge drift across future Haiku model updates.

### Cost guard

Each case costs roughly:
- Input ~150 tokens × Haiku input rate
- Output ~80 tokens × Haiku output rate

Twenty cases per full run total ≈ 3K input + 1.6K output tokens, which on
Claude Haiku 4.5 prices to roughly **$0.01 per full run** — comfortably
below the $1.00 ceiling.

The script will print an estimated cost before making any API call and
abort if the estimate exceeds `--max-cost` (default $1.00). Estimation
uses Anthropic's `count_tokens` endpoint where available, with a
4-chars-per-token approximation as a fallback. The script also requires
`ANTHROPIC_API_KEY` to be set and exits with a clear message if it isn't.

### Per-detector accept criteria (placeholder for future calibration)

Not a release gate now — written here so the eventual fixture-writing
session has a target:

| Detector                       | Min precision | Min recall | Notes |
|--------------------------------|---------------|------------|-------|
| FM-2.6 ReasoningActionMismatch | 0.70          | 0.60       | False-positive cost is high (steers a correct agent), so precision is weighted heavier than recall. |
| FM-3.1 PrematureTermination    | 0.60          | 0.70       | False-negative cost is high (lets bad output ship), so recall is weighted heavier than precision. |

These numbers are placeholders. The first eval run sets the actual
baseline; the threshold should be tightened only after we've seen what
the prompt produces on realistic inputs.

---

## What this audit does not cover

- The non-LLM detectors (StepRepetition, UnawareTermination,
  NoProgressLoop) have automated tests but no measured precision/recall
  against real agent traces. That is the work of Item 4 (real LangGraph
  run) in this same direction-correction pass, and a follow-up MAST-Data
  validation pass not scheduled here.
- Judge drift as new Anthropic models ship is not actively monitored. The
  harness's per-run JSON output is designed to enable drift tracking, but
  no policy is set for when to re-run.
- Adversarial inputs (prompt-injection attempts in agent reasoning that
  try to steer the judge's verdict) are not in the proposed fixture set.
  That belongs in a separate threat-model exercise, not in baseline
  calibration.

---

*Audit date: 2026-05-19. Refers to ProcessGuard v0.1.0–v0.1.1 code.*
