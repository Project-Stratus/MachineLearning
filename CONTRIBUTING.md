# Contributing to Project Stratus

## Quickstart Checklist
- Follow **Getting Started** steps in README
- Run `pytest` before pushing to confirm nothing regresses.
- Describe your changes and testing in every pull request.

## Where things are documented
- `README.md` — setup and how to run things.
- `CLAUDE.md` — project scope, constraints, and domain terms (ZP/SP, station-keeping, etc).
- `notes/development_roadmap.md` — the staged plan (Layer 1-5) and why things are sequenced that way.
- `todo.md` — small tactical chores.

Check these before asking — most "why is it like this" questions are answered in CLAUDE.md or the roadmap.

## Development Workflow
1. Discuss major changes before writing code.
2. Keep pull requests focused; prefer multiple small PRs to one large one.
3. Ensure commits are atomic and have descriptive messages.
4. Discuss when PRs to main are ready

## Testing Expectations
- Add/extend unit tests in `tests/` for new functionality.
- Update fixtures or test data when you change interfaces.
- For training changes, document how you validated learning performance.
- Baselines are dim-specific (see CLAUDE.md) — check against the right dimension's numbers before calling a result good. 1D is useful for debugging but isn't a meaningful test on its own.

## Documentation
- Update README.md or module docstrings when behaviour changes.

## Communication
- Use WhatsApp for most communication and project management
