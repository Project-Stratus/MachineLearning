"""Regenerate the exercise notebooks from the solution notebooks.

The solutions in ``onboarding/solutions/`` are the source of truth. Edit those,
then run this from anywhere:

    python onboarding/make_exercises.py

For each solution notebook it writes an exercise copy to ``onboarding/`` with:

* code between ``# BEGIN SOLUTION`` and ``# END SOLUTION`` replaced by
  ``# YOUR CODE HERE`` and ``raise NotImplementedError``, at the same indent;
* markdown between ``<!-- BEGIN SOLUTION -->`` and ``<!-- END SOLUTION -->``
  replaced by a prompt to write an answer;
* cells tagged ``solution`` removed entirely;
* all outputs cleared and cell ids renumbered.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE / "solutions"

CODE_BEGIN = "# BEGIN SOLUTION"
CODE_END = "# END SOLUTION"
MD_SOLUTION = re.compile(
    r"<!-- BEGIN SOLUTION -->.*?<!-- END SOLUTION -->", flags=re.DOTALL
)
MD_PROMPT = "*✏️ Write your answer here.*"


def strip_code(source: str, where: str) -> str:
    out: list[str] = []
    inside = False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == CODE_BEGIN:
            if inside:
                raise ValueError(f"{where}: nested {CODE_BEGIN}")
            inside = True
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# YOUR CODE HERE")
            out.append(f"{indent}raise NotImplementedError")
        elif stripped == CODE_END:
            if not inside:
                raise ValueError(f"{where}: {CODE_END} without {CODE_BEGIN}")
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError(f"{where}: {CODE_BEGIN} is never closed")
    return "\n".join(out)


def make_exercise(solution_path: Path) -> Path:
    nb = nbformat.read(solution_path, as_version=4)
    cells = []
    for i, cell in enumerate(nb.cells):
        if "solution" in cell.get("metadata", {}).get("tags", []):
            continue
        where = f"{solution_path.name} cell {i}"
        if cell.cell_type == "code":
            cell.source = strip_code(cell.source, where)
            cell.outputs = []
            cell.execution_count = None
        elif cell.cell_type == "markdown":
            cell.source = MD_SOLUTION.sub(MD_PROMPT, cell.source)
        cells.append(cell)
    # Number cell ids 0, 1, 2, ... as the nbstripout pre-commit hook does, so
    # regenerating doesn't leave changes for the hook to make.
    for i, cell in enumerate(cells):
        cell.id = str(i)
    nb.cells = cells
    target = HERE / solution_path.name
    nbformat.write(nb, target)
    return target


def main() -> None:
    for path in sorted(SOLUTIONS.glob("*.ipynb")):
        print(f"{path.relative_to(HERE)} -> {make_exercise(path).relative_to(HERE)}")


if __name__ == "__main__":
    main()
