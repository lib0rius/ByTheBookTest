"""File format of the dataset: one JSON object per line, validated with Pydantic.

A dataset folder holds ``kbs.jsonl`` (one rulebook per line, as :class:`KBRecord`) and ``claims.jsonl``
(one claim per line, as :class:`ClaimRecord`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Expected = Literal["TRUE", "FALSE", "UNKNOWN"]
"""The three possible answers: the claim follows, its negation follows, or neither follows."""


class StatementRecord(BaseModel):
    """One statement of a rulebook."""

    id: str
    text: str
    """The English sentence shown to the model."""
    form: str
    """The logical form family the sentence was rendered from (for example ``"rule"``)."""
    logic: str | None
    """The formula checked by Z3, as text; ``None`` for statements that carry no logic."""
    role: Literal["proof", "trap", "noise"]
    """``proof``: needed by some claim; ``trap``: close to a proof but not enough; ``noise``: unrelated."""
    world: Literal["conflict", "agree", "neutral"]
    """Relation with the real world: contradicts it, agrees with it, or uses invented words only."""
    category: str | None
    """The fact table the statement comes from, if any."""


class KBRecord(BaseModel):
    """A rulebook (knowledge base): the only source of truth for its claims."""

    kb_id: str
    level: int
    statements: list[StatementRecord]
    """In the order they are shown to the model."""


class ClaimRecord(BaseModel):
    """A claim to judge against one rulebook, with its expected answer and how it was built."""

    claim_id: str
    kb_id: str
    level: int
    group: str
    """The kind of reasoning needed (for example ``"rule_chain"``)."""
    subtype: str
    """The variant of the group the claim was built from."""
    text: str
    logic: str
    expected: Expected
    z3: Expected
    """The answer computed independently by Z3; always equal to ``expected`` in a released dataset."""
    world_relation: Literal["conflict", "agree", "unsupported_true", "unsupported_false", "neutral"]
    """How the expected answer relates to the real world. ``unsupported_true`` and ``unsupported_false``
    are UNKNOWN claims that are true or false in the real world: the rulebook is silent about them."""
    category: str | None
    depth: int
    """Number of reasoning steps from the rulebook to the answer."""
    reasoning: list[str]
    """The reasoning steps, in words."""
    support: list[str]
    """Ids of the statements the answer depends on."""
    traps: list[str]
    """Ids of the trap statements built for this claim."""
