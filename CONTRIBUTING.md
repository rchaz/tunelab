# Contributing to tunelab

## Quick links

- [Issues](https://github.com/rchaz/tunelab/issues)
- [README](README.md)

## Reporting bugs

Open an issue with:

1. What you expected to happen
2. What actually happened
3. Steps to reproduce
4. Your environment (OS, Python version, `uv --version`)

## Suggesting features

Open an issue describing your use case and proposed solution. For larger changes, open a discussion first.

## Development setup

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/), Apple Silicon Mac (for training tests only — decide/data/eval tests run anywhere).

```bash
git clone https://github.com/rchaz/tunelab.git
cd tunelab
```

No `pip install` needed — every script uses uv with inline dependency declarations.

## Running tests

```bash
bash tests/run_all.sh
```

First run downloads ~500MB of models into `~/.cache/huggingface` and caches uv dependencies; later runs are fast.

Individual tests:

```bash
python3 tests/test_centroid_classify.py
python3 tests/test_split_data.py
# etc.
```

## Project structure

```
skills/           # The five Claude Code skills (SKILL.md + scripts/)
  tune-decide/    # Front door — interviews, experiments, recommends
  tune-data/      # Dataset building, cleaning, splitting
  tune-train/     # Local LoRA/QLoRA/CPT training (MLX)
  tune-eval/      # Evaluation, cascade composition, LLM-as-judge
  tune-loop/      # Continuous improvement loop
concepts/         # Plain-English explainers for every idea tunelab uses
recipes/          # Worked end-to-end examples with real numbers
dogfood/          # Internal test runs and benchmark results
tests/            # Script-level tests (subprocess, real data)
```

## Pull request guidelines

1. Keep PRs focused — one change per PR
2. `bash tests/run_all.sh` must pass
3. Follow existing code patterns
4. No new dependencies in scripts unless absolutely necessary (and declared inline via uv)
5. If you add a new script, add a test for it

## Commit messages

Use clear, descriptive commit messages in imperative style.

## AI-assisted contributions

AI-assisted PRs are welcome. If you used AI tools, mention it in the PR description for transparency.

## Security

If you discover a security vulnerability, please report it privately. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
