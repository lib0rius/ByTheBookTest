"""Reading the final answer of a chat model.

Two readings are recorded for every response:

- **exact**: after removing the thinking and surrounding spaces, the answer is valid only if it is a JSON
  object with the single field ``answer`` holding one of the three labels (checked by Pydantic);
- **lenient**: also accepts the JSON inside a code block or with other text around it, and the bare word,
  as long as exactly one label is given.

Decision models (``kind = "systemone"``) return a structured choice instead and do not use this module.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, ValidationError

from bythebooktest.records import Expected

LABELS = ("TRUE", "FALSE", "UNKNOWN")
_THINK_END = "</think>"
_OBJECT = re.compile(r"\{[^{}]*\}")
_WORD = re.compile(r"\b(TRUE|FALSE|UNKNOWN)\b")


class FinalAnswer(BaseModel):
    """The only answer format accepted by the exact reading."""

    model_config = ConfigDict(extra="forbid", strict=True)
    answer: Expected


def strip_thinking(content: str) -> str:
    """Remove a thinking block left in the text, when the server did not move it to ``reasoning_content``."""
    if _THINK_END in content:
        content = content.rsplit(_THINK_END, 1)[1]
    return content.strip()


def parse_exact(content: str) -> Expected | None:
    """Return the label if the response is exactly ``{"answer": "<label>"}``, else ``None``."""
    try:
        return FinalAnswer.model_validate(json.loads(strip_thinking(content))).answer
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def parse_lenient(content: str) -> Expected | None:
    """Return the label if the response names exactly one, in JSON or as a bare word, else ``None``."""
    exact = parse_exact(content)
    if exact is not None:
        return exact
    text = strip_thinking(content)
    values: set[str] = set()
    for m in _OBJECT.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("answer") in LABELS:
            values.add(obj["answer"])
    if not values:
        values = set(_WORD.findall(text))
    return values.pop() if len(values) == 1 else None  # type: ignore[return-value]
