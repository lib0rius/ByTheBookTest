"""Invented words for the neutral part of a rulebook: names of entities and of classes.

Words have three syllables, which makes it unlikely that they match an English word or a real name and
so bring world knowledge with them. Every word is unique within a dataset generated with one vocabulary.
"""

from __future__ import annotations

import random

ONSETS = [
    "b",
    "d",
    "f",
    "g",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "z",
    "br",
    "dr",
    "fl",
    "gr",
    "kr",
    "pr",
    "sk",
    "sp",
    "st",
    "tr",
    "zh",
]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["l", "n", "r", "s", "k", "x", "t"]


class Vocabulary:
    """Draws unique invented words from a seeded random generator.

    Args:
        rng: The generator; the same seed gives the same words in the same order.
        reserved: Words that must never be produced (compared in lower case).
    """

    def __init__(self, rng: random.Random, reserved: set[str] | None = None) -> None:
        self.rng = rng
        self.used: set[str] = {w.lower() for w in (reserved or set())}

    def _word(self, syllables: int) -> str:
        parts = [self.rng.choice(ONSETS) + self.rng.choice(VOWELS) for _ in range(syllables)]
        return "".join(parts) + self.rng.choice(CODAS)

    def word(self, syllables: int = 3) -> str:
        """Return a new lower-case word, never returned before."""
        for _ in range(10000):
            w = self._word(syllables)
            if w not in self.used:
                self.used.add(w)
                return w
        raise RuntimeError("vocabulary exhausted")

    def name(self) -> str:
        """Return a proper name for an invented entity, for example ``"Kelvorax"``."""
        return self.word().capitalize()

    def noun(self) -> str:
        """Return a common noun for an invented class, for example ``"trusapil"``."""
        return self.word()
