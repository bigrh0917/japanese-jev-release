"""Text template for the three primitives.

Training (data_pipeline/build.py), inference (readout.py) and serving (serve.py) all take render() from here, so
the text seen in training is exactly the text built at inference.

    [kind] choice
    [state] <the situation>
    [question] <the question>
    [options]
    A. <option name>: <description>
    B. ...
    [decision]:

The completion is one letter with a leading space, e.g. " B".
"""

from __future__ import annotations

KINDS = ("choice", "noul", "score")
LETTERS = "ABCDEFGHIJKLMNOP"  # at most 16 options


def render(kind: str, state: str, question: str, options: list[tuple[str, str]]) -> str:
    """options is a list of (name, description); the description may be an empty string."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"the number of options must be between 2 and {len(LETTERS)}, got {len(options)}")
    lines = [f"[kind] {kind}", f"[state] {state}", f"[question] {question}", "[options]"]
    for i, (name, desc) in enumerate(options):
        lines.append(f"{LETTERS[i]}. {name}: {desc}" if desc else f"{LETTERS[i]}. {name}")
    lines.append("[decision]:")
    return "\n".join(lines)
