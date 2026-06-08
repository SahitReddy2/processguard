# LLM-judge calibration

This document records how well the two LLM-judge detectors shipped in
v0.1.1 — `ReasoningActionMismatchDetector` (FM-2.6) and
`PrematureTerminationDetector` (FM-3.1) — agree with a hand-labeled
set of 20 calibration cases. The number reported here is the single
deliverable that distinguishes v0.3 from a release that "shipped an
LLM judge with no idea whether it works."

> **Headline (pending freeze-labels commit):**
> κ = **TBD** (N=20, single labeler) — measurement scheduled after
> the `freeze calibration labels (v0.3, N=20)` commit lands.

The headline number, wherever it appears in this repo (this doc, the
README, release notes), is reported with N and labeler-count inline
per the v0.3 plan §3.6 reporting discipline. It is not allowed to
travel without the caveat.

---

## What this measures

Cohen's kappa between two raters:

- **Rater 1 (human)**: the project maintainer, labeling each case
  before seeing the judge's output. Labels are frozen in a dedicated
  git commit before the calibration script first runs.
- **Rater 2 (deployed judge)**: the FM-2.6 / FM-3.1 detector's `_judge()`
  method, run with the production prompt against Claude Haiku 4.5 at
  the default `confidence_floor=0.5`.

A "fire" verdict means the detector said the failure mode is present:

- FM-2.6 fire = **MISMATCH**, no-fire = MATCH.
- FM-3.1 fire = **INCOMPLETE**, no-fire = COMPLETE.

The detector is treated as a black box: whatever the deployed prompt
returns through the deployed parsing path is what's measured. This is
the same code that runs in production. We are not measuring the raw
LLM — we are measuring what ProcessGuard actually does.

---

## Why kappa and not just accuracy

Both are reported. Accuracy is more legible; kappa corrects for chance
agreement and is the metric a reviewer who actually does eval work will
look for first.

A 50/50 marginal distribution with random independence has p_e = 0.5,
so a judge that achieved 60% raw accuracy would have only κ ≈ 0.2 —
informative for headline framing, since the judge isn't doing *that*
much better than a coin. Accuracy alone hides this.

Both numbers appear inline. Per-class precision/recall/F1 are reported
alongside for cases where false-positive cost (steers a correct agent)
and false-negative cost (lets bad output ship) are weighted
differently.

---

## Methodology

### Case drafting

The 30 candidate scenarios (15 per detector) were drafted by Claude
during v0.3 Phase B. Scenarios were drafted to span the failure
categories named in [`judge_audit.md`](judge_audit.md) proportionally,
not weighted toward "tricky" cases. The intent is to measure how the
prompt handles each category, not how it handles one carefully crafted
adversarial example.

Categories targeted (RAM):

- 5 multi-step-plan cases (false-positive risk: the judge flags
  step-1-of-3 actions as MISMATCH)
- 4 semantic-drift cases (false-negative risk: same tool, drifted
  intent)
- 3 truncation-interaction cases (the controlling intent lives in the
  tail of >500-char reasoning that gets clipped)
- 3 clean-match controls

Categories targeted (PT):

- 5 confident-tone-incomplete cases (false-negative risk: paragraph
  shape suggests "finished" but specific deliverables are missing)
- 4 terse-correct cases (false-positive risk: short factual answer to
  a fact-shaped question)
- 3 truncation-interaction cases (task and/or output >400 chars; the
  judge sees only the leads)
- 2 clean-complete controls
- 1 clean-incomplete control

### Labeling

The maintainer picked 10 of 15 candidates per detector and assigned an
expected verdict to each. Selection and labeling happen in a single
session of 3-4 hours (or two consecutive sittings within 24 hours) to
reduce intra-labeler drift, per the v0.3 plan §5 cadence note.

The categories file (`datasets/calibration/categories.jsonl`) is **not
loaded by the labeling workflow** — the labeler sees only id +
reasoning/task + action/output. Categories are only joined back at
report-generation time. This prevents subconscious label-to-category
matching during labeling, which would corrupt the per-category
breakdown's meaning.

Cases that the labeler could not decisively label in 60 seconds were
excluded via the ambiguity protocol (moved back to `*_candidates_v1`
with `rejected_for_ambiguity: true`), not force-labeled. Force-labels
are more harmful than excluded labels for calibration.

After labeling, the labels were committed in a dedicated git commit
titled `freeze calibration labels (v0.3, N=20)` before the
calibration script was first executed.

### Running the calibration

After the freeze commit landed, `scripts/run_calibration.py` was run
with `ANTHROPIC_API_KEY` set. The script:

1. Loaded `ram_v1.jsonl` and `pt_v1.jsonl` (refusing to run if any
   `expected_verdict` was null — a safety net for the unfrozen-draft
   case).
2. For each case, constructed the events the detector expects
   (REASONING + TOOL_CALL for FM-2.6; MESSAGE + TERMINATE + set_task
   for FM-3.1), wrapped the Anthropic client in a recording proxy that
   captures the raw response text for every call (including the
   no-fire path where the detector returns None — without the proxy,
   that response text would be lost), and ran the deployed detector's
   `observe()` chain.
3. Recorded one `Observation` per case: expected verdict, observed
   verdict, raw judge text, confidence.
4. Computed the confusion matrix, accuracy, P/R/F1, kappa, and
   per-category breakdown.

The script wrote machine-readable JSON to
`calibration_results/{ram,pt}_v0.3_<UTC>.json` and a human-readable
markdown report alongside it. The headline numbers from each run were
manually inserted into this document (deliberately a manual step —
the public artifact gets a human review before it lands).

### Post-judge-review label flips

If, after running the calibration, the maintainer reviewed a
disagreement on their own and decided their label was wrong, the flip
happened by hand in a dedicated commit titled
`relabel calibration case <id> (v0.3, post-judge-review)` and the
calibration was re-run from scratch. The total flip count for v0.3
appears below in the headline section.

Per v0.3 plan §8, Claude does not adjudicate label-judge
disagreements. Asking the agent "does my label or the judge's verdict
seem more correct?" would re-introduce the dependency the methodology
was designed to avoid.

---

## Known limitations

- **N = 20 is small.** Confidence intervals on kappa with N=20 are
  wide. The headline is internal-baseline-grade, not paper-grade. A
  v0.4 release may grow N or add inter-annotator agreement (a second
  labeler on the same set, giving a kappa ceiling for what the judge
  could realistically hit).
- **Single labeler.** No inter-annotator agreement is computed. The
  number reported is "agreement between one human and the deployed
  judge", not "agreement between humans" or "agreement among multiple
  judges."
- **Prompts not tuned.** The FM-2.6 and FM-3.1 prompts shipped in
  v0.1.1 are what's measured. If the numbers come back low, the next
  step is prompt iteration in v0.4, not a re-label of the same cases
  to make the prompt look better.
- **No bootstrap or significance test.** The kappa point estimate is
  reported without a confidence interval. For N=20 the rough 95% CI
  on a kappa of 0.5 is roughly ±0.3 (using the Fleiss-Cohen
  approximation); a future release with larger N would warrant
  computing this properly.
- **Categories are author-assigned.** I (Claude) chose which category
  each scenario was drafted to exercise. The labeler did not see those
  assignments while labeling, which removes the subconscious-matching
  failure mode, but the categories themselves are still author-defined
  buckets.

---

## FM-2.6 — ReasoningActionMismatch

_Filled in after the calibration script runs against the freeze-labels
commit. The full per-case verdict table from the run is committed to
`calibration_results/ram_v0.3_<UTC>.md` and excerpted here._

```
κ = TBD (N=10, single labeler)
raw accuracy = TBD
precision = TBD • recall = TBD • F1 = TBD
```

Per-category breakdown: TBD.

Per-case verdicts: TBD.

---

## FM-3.1 — PrematureTermination

_Filled in after the calibration script runs against the freeze-labels
commit. Same shape as FM-2.6 above._

```
κ = TBD (N=10, single labeler)
raw accuracy = TBD
precision = TBD • recall = TBD • F1 = TBD
```

Per-category breakdown: TBD.

Per-case verdicts: TBD.

---

## Reproducing

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_calibration.py --detector ram
python scripts/run_calibration.py --detector pt
python scripts/run_calibration.py --all      # both detectors in one run
```

Outputs land in `calibration_results/`. The reports include the full
per-case verdict log including raw judge text — so a reviewer can
spot-check any disagreement without re-running the live judge.

`calibration_results/` is gitignored. The headline numbers move from a
report file into this document by hand.

---

## Cost

- ~20 cases × ~150 input + ~80 output tokens × Claude Haiku 4.5 rates
  ≈ $0.05 per full run.
- Expect 2-3 runs across iteration (catch a bug, re-run; tweak fixture,
  re-run). Total: $0.10–$0.20.
- The Anthropic key used for calibration should have a $5/month spend
  cap configured at the provider as the real safety net. Per-script
  cost estimation is not the primary defence.

---

*Calibration doc scaffold authored 2026-06-07 as part of v0.3 Phase B.
Last actual measurement: pending freeze commit.*
