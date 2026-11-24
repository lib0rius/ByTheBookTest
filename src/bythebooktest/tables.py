"""The curated tables of well-known content, and their validation.

- ``data/tables/facts/<category>.toml``: facts about very well-known subjects, one file per category: the
  true version, some false versions and, for implicit assumptions, statements that are true only under a
  written condition (``traps``).
- ``data/tables/abilities.toml``: well-known classes and actions, with what holds in the real world for
  "can" and "does".

A predicate is written as a third-person-singular phrase ("was born in Corsica"); its negation is derived
for regular forms ("was not born in Corsica", "does not orbit the Earth"). Where the rule is not enough,
the pair is written in full: ``["never meets", "meets somewhere"]``.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["history_geography", "nature", "math_geometry", "common_sense", "implicit_assumptions"]

_AUXILIARIES = {"is", "was", "are", "were", "will", "must", "should", "would"}
_IRREGULAR_THIRD = {"has": "have", "does": "do", "goes": "go", "is": "be"}


def third_to_base(word: str) -> str:
    """Base form of a third-person-singular verb: "flies" -> "fly", "watches" -> "watch"."""
    if word in _IRREGULAR_THIRD:
        return _IRREGULAR_THIRD[word]
    if word.endswith("ies"):
        return word[:-1] if len(word) <= 4 else word[:-3] + "y"  # "dies" -> "die", "flies" -> "fly"
    if word.endswith(("ches", "shes", "sses", "xes", "zzes", "oes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    raise ValueError(f"'{word}' is not a regular third-person verb: write the negation in full")


_NEGATIVE_WORDS = {"no", "not", "never", "nothing", "none", "nobody", "cannot"}


def negate(vp: str) -> str:
    """Negation of a regular verb phrase.

    A phrase that already contains a negative word ("has no rings") needs its negation written in full,
    because the rule would give a double negation ("does not have no rings").
    """
    if _NEGATIVE_WORDS & set(vp.lower().split()):
        raise ValueError(f"'{vp}' contains a negation: write the pair in full")
    first, _, rest = vp.partition(" ")
    rest = f" {rest}" if rest else ""
    if first == "can":
        return f"cannot{rest}"
    if first in _AUXILIARIES:
        return f"{first} not{rest}"
    if first == "had":
        return f"did not have{rest}"
    return f"does not {third_to_base(first)}{rest}"


class VP(BaseModel):
    """A third-person-singular verb phrase, with its negation."""

    pos: str = Field(min_length=1)
    neg: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, value: object) -> object:
        if isinstance(value, str):
            return {"pos": value, "neg": negate(value)}
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"pos": value[0], "neg": value[1]}
        return value


class FactRow(BaseModel):
    """A well-known fact: its subject, the true predicate, false ones and conditional traps."""

    id: str
    category: Category
    subject: str
    kind: Literal["person", "thing"] = "thing"
    true: VP
    false: list[VP] = Field(min_length=1)
    traps: list[VP] = Field(default_factory=list)

    @model_validator(mode="after")
    def _distinct_vps(self) -> FactRow:
        texts = [self.true.pos] + [v.pos for v in self.false] + [v.pos for v in self.traps]
        if len(set(texts)) != len(texts):
            raise ValueError(f"{self.id}: repeated predicates in the same row")
        return self


class AbilityRow(BaseModel):
    """A well-known class and action, with what holds in the real world for "can" and "does"."""

    id: str
    category: Category
    singular: str
    plural: str
    verb: str
    third: str
    world_can: bool
    world_does: bool


class Tables(BaseModel):
    """All the tables, with the fingerprint of their files (recorded in every dataset manifest)."""

    facts: list[FactRow]
    abilities: list[AbilityRow]
    sha256: str

    @model_validator(mode="after")
    def _unique_ids(self) -> Tables:
        ids = [r.id for r in self.facts] + [r.id for r in self.abilities]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"repeated ids in the tables: {sorted(dup)}")
        return self

    def reserved_words(self) -> set[str]:
        """Words used in the tables: the invented vocabulary must never produce them."""
        words: set[str] = set()
        texts = []
        for r in self.facts:
            texts += [r.subject, r.true.pos] + [v.pos for v in r.false + r.traps]
        for a in self.abilities:
            texts += [a.singular, a.plural, a.verb, a.third]
        for t in texts:
            words.update(w.strip(".,").lower() for w in t.split())
        return words


def _rows(path: Path, digest: hashlib._Hash) -> list[dict]:
    raw = path.read_bytes()
    digest.update(path.name.encode() + b"\0" + raw)
    data = tomllib.loads(raw.decode("utf-8"))
    rows = data.get("row", [])
    for r in rows:
        if "category" in data:
            r.setdefault("category", data["category"])
    return rows


def load_tables(directory: Path) -> Tables:
    """Read and validate every table in ``directory`` (``facts/*.toml`` and ``abilities.toml``)."""
    digest = hashlib.sha256()
    facts: list[dict] = []
    for path in sorted((directory / "facts").glob("*.toml")):
        facts += _rows(path, digest)
    abilities = _rows(directory / "abilities.toml", digest)
    return Tables.model_validate({"facts": facts, "abilities": abilities, "sha256": digest.hexdigest()})
