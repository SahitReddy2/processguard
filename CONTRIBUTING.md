# Contributing to ProcessGuard

Thanks for considering a contribution. ProcessGuard is small enough that one
pull request can meaningfully move it forward — and small enough that the
review bar is real. Please read this end-to-end before opening a non-trivial
PR.

## Setup

```bash
git clone https://github.com/SahitReddy2/processguard.git
cd processguard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[langgraph,dev]"
```

## Running things

| Command | What it does |
|---|---|
| `pytest -q` | Run the test suite (should be fast — ≤ 1s on v0.2). |
| `python examples/synthetic_raw_loop_demo.py` | No-LLM demo of the rule-based detectors firing. |
| `python examples/synthetic_langgraph_demo.py` | LangGraph synthetic demo. Requires `ANTHROPIC_API_KEY`. |
| `python examples/real_langgraph_demo.py` | LangGraph real-agent demo. Requires `GOOGLE_API_KEY`. |
| `python scripts/run_eval.py --gold datasets/gold/v0.2.jsonl` | Run the offline eval harness (the same script CI runs). |

## Code style

- Python ≥ 3.10. Type hints on public APIs. `from __future__ import annotations` everywhere.
- No `print` in library code (`processguard/`); only in `examples/` and `scripts/`.
- Detector docstrings are five-sentence contracts (what / when fires / smallest case / must-not-fire / known limitation). See existing detectors for the shape.
- Tests live in `tests/`. New behaviour gets a test before the implementation if it's a regression for a real bug; tests-after is fine for net-new features.

## Pull request process

1. **Open an issue first** if the change is more than a small fix. It saves both of us time if the direction needs adjustment.
2. **Run the eval gate locally** before pushing: `python scripts/run_eval.py --gold datasets/gold/v0.2.jsonl`. CI runs the same command; failing locally and pushing wastes a CI minute.
3. **Keep PRs scoped.** One logical change per PR. Refactors and feature work in separate PRs.
4. **Commit messages** describe the *why* in one line, plus a body if the change isn't obvious from the diff. No `Co-Authored-By: ` trailers.

## CI gate

Every PR triggers two GitHub Actions:

- **`tests`** — runs pytest on Python 3.10 / 3.11 / 3.12.
- **`eval-gate`** — runs `scripts/run_eval.py` against `datasets/gold/v0.2.jsonl` and posts a markdown report as a PR comment. Deterministic cases must pass; LLM-required cases skip in CI (no API keys) and that counts as build-passing.

The `main` branch should be configured (via repo Settings → Branches → Branch
protection rules) to require both workflows to pass before merge. This is a
manual repo-settings step; it is not enforced by the workflow files themselves.
Enable it once the project has external contributors — for a solo repo it
mostly creates friction.

## Detector contracts

The five v0.1.1 detectors have explicit semantic contracts in their docstrings.
If you add a new detector, write the same five-sentence contract in plain
English at the top of the class:

1. What failure mode it identifies.
2. When it fires (semantic claim — no thresholds, no implementation terms).
3. The smallest meaningful failure case it should catch (concrete example).
4. A case where it must NOT fire (concrete counter-example).
5. Its known limitation (one honest sentence).

See `processguard/detectors/step_repetition.py` for the template.

## License

By contributing, you agree that your contributions are licensed under the same
[MIT license](LICENSE) as the project.
