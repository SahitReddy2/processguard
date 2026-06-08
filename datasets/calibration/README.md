# Calibration set — v0.3

This directory holds the artifacts the v0.3 calibration uses. The layout
is intentional and the protocol around it is the load-bearing part of
v0.3's methodology.

## Files

| File | Who edits | When |
|---|---|---|
| `ram_candidates_v1.jsonl` | Claude | At draft time (Phase B, this PR) |
| `pt_candidates_v1.jsonl` | Claude | At draft time (Phase B, this PR) |
| `categories.jsonl` | Claude | At draft time. **Not loaded by the labeling workflow.** |
| `ram_v1.jsonl` | **Maintainer (you), by hand** | At freeze time, after labeling |
| `pt_v1.jsonl` | **Maintainer (you), by hand** | At freeze time, after labeling |

## Why categories live in a separate file

Each candidate scenario in `*_candidates_v1.jsonl` has only the fields
needed to label it: `id`, the reasoning/task text, the action/output
text, and a null `expected_verdict` to fill in.

The category each scenario was drafted to exercise (multi-step-plan,
semantic-drift, truncation-interaction, etc.) is stored separately in
`categories.jsonl`, keyed by id. The labeler does not see this file
during labeling.

If the category were inside the candidate file, two failure modes
become available to the labeler:

1. Subconsciously matching labels to categories ("this is a
   multi-step-plan case, so the label is MATCH"). The per-category
   breakdown in the calibration report then measures pattern-matching,
   not judgment.
2. Mentally drifting toward whichever verdict the category name
   suggests for ambiguous cases.

Both corrupt the calibration in ways that don't show up in any single
metric. The split-file design makes the corruption mechanically harder
to introduce.

## Labeling protocol (when the time comes)

1. **Read all 30 candidates first**, top to bottom, without labeling
   anything. Get a sense of the distribution.
2. **Pick your 10 strongest per detector.** Don't label yet.
3. **Label, one at a time, fast.** First instinct, written down. If a
   case takes more than 60 seconds to decide, exclude it (ambiguity
   protocol — see `docs/v0.3_plan.md` §5).
4. **After labeling, do not re-edit.** No "let me double-check that
   one." The freeze commit is the freeze commit.

Commit title: `freeze calibration labels (v0.3, N=20)`. No other
changes in that commit.

After labels are frozen and pushed, the calibration script runs.
`categories.jsonl` is read at report-generation time only, to build
the per-category breakdown.

## Format

### Candidate files (`*_candidates_v1.jsonl`)

RAM (one JSON object per line):

```json
{
  "id": "ram-001",
  "reasoning": "I'll search for the latest papers, then read them, then summarise.",
  "tool_name": "web_search",
  "tool_args": {"query": "..."},
  "expected_verdict": null
}
```

PT:

```json
{
  "id": "pt-001",
  "task": "Find 5 papers on X and summarise each.",
  "output": "X has emerged as a strong technique...",
  "expected_verdict": null
}
```

### Frozen files (`*_v1.jsonl`)

Identical shape to the candidates, but only 10 entries each (the 20
you picked) and `expected_verdict` filled in:

- RAM: `"match"` or `"mismatch"`.
- PT: `"complete"` or `"incomplete"`.

### Categories file (`categories.jsonl`)

One JSON object per line, `{id, category}`. Used by the calibration
report-generation step, not by labeling.

## What this is NOT

- Not a tutorial dataset. The scenarios are drafted to exercise the
  judge prompts; they are not designed to teach anyone what a "good"
  agent looks like.
- Not a benchmark. N=20, single labeler, no inter-annotator agreement.
  The calibration number is internal-baseline-grade, not paper-grade.
- Not curated for difficulty. Scenarios span the named categories
  proportionally; some are obvious, some are subtle. That's the point.
