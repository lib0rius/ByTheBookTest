"""Planning the claims: how many, with which expected answer, relation with the world and depth.

Every cell (one level times one reasoning group) gets the same number of claims, split among the
rulebooks of the level. Inside every cell the claims follow the same mix of expected answers and world
relations (:data:`CLAIM_MIX`), and depths and variants rotate evenly, so that each depth and each variant
keeps that mix.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .logic import Verdict

CLAIM_MIX: list[tuple[Verdict, str, int]] = [
    (Verdict.TRUE, "conflict", 18),
    (Verdict.TRUE, "agree", 5),
    (Verdict.TRUE, "neutral", 10),
    (Verdict.FALSE, "conflict", 18),
    (Verdict.FALSE, "agree", 5),
    (Verdict.FALSE, "neutral", 10),
    (Verdict.UNKNOWN, "unsupported_true", 12),
    (Verdict.UNKNOWN, "unsupported_false", 12),
    (Verdict.UNKNOWN, "neutral", 10),
]
"""Claims per 100, by expected answer and relation with the real world.

Conflict claims are the majority because they are what the benchmark is about; ``agree`` claims are the
comparison for the conflict effect; ``neutral`` claims use invented words only.
"""

LEVEL_DEPTHS = {1: (0, 1), 2: (1, 3), 3: (2, 6)}
"""Allowed number of reasoning steps per level (short, medium and long rulebooks)."""


@dataclass(frozen=True)
class ClaimSpec:
    """What a claim must be, before it is built."""

    group: str
    expected: Verdict
    relation: str
    depth: int
    variant: str = ""
    """Scheme inside the group, for example ``"forward"`` or ``"backward"``."""


def scaled_counts(n: int) -> list[tuple[Verdict, str, int]]:
    """Scale :data:`CLAIM_MIX` to ``n`` claims, rounding by the largest-remainder method."""
    raw = [(v, r, c * n / 100) for v, r, c in CLAIM_MIX]
    counts = [(v, r, int(x)) for v, r, x in raw]
    missing = n - sum(c for _, _, c in counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i][2] - int(raw[i][2]), reverse=True)
    for i in order[:missing]:
        v, r, c = counts[i]
        counts[i] = (v, r, c + 1)
    return counts


def cell_specs(
    group: str, n: int, depths: list[int], rng: random.Random, variants: list[str] | None = None
) -> list[ClaimSpec]:
    """Plan the ``n`` claims of one cell.

    Within every (expected answer, relation) block, variants and depths rotate evenly, so every variant
    and every depth gets the same mix of answers and relations.
    """
    variants = variants or [""]
    specs: list[ClaimSpec] = []
    start_v = rng.randrange(len(variants))
    g = 0  # variants rotate over the whole cell, so they stay balanced even with few claims
    for verdict, relation, count in scaled_counts(n):
        start_d = rng.randrange(len(depths))
        for i in range(count):
            variant = variants[(start_v + g) % len(variants)]
            depth = depths[(start_d + i) % len(depths)]
            specs.append(ClaimSpec(group, verdict, relation, depth, variant))
            g += 1
    return specs


def split_among_kbs(specs: list[ClaimSpec], n_kbs: int, rng: random.Random) -> list[list[ClaimSpec]]:
    """Split the claims of a cell among the rulebooks of the level, keeping the mix in each rulebook."""
    ordered = sorted(specs, key=lambda s: (s.expected, s.relation, s.variant, s.depth))
    offset = rng.randrange(n_kbs)
    buckets: list[list[ClaimSpec]] = [[] for _ in range(n_kbs)]
    for i, spec in enumerate(ordered):
        buckets[(i + offset) % n_kbs].append(spec)
    return buckets
