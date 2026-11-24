"""Reasoning groups, and the proof schemes each claim is built from.

A group receives a :class:`~bythebooktest.plan.ClaimSpec` (expected answer, relation with the real
world, depth), adds to the rulebook the statements the claim needs, and records the claim. The expected
answer set here is the one intended by construction; the generator then checks it with Z3.

Level 1 groups (this module):

- ``direct_fact``: the claim, or its negation, is written in the rulebook (depth 0);
- ``rule_chain``: a chain of simple rules from the subject to the claim;
- ``modal``: "generally", "can", "must", "is allowed to".

Level 2 and Level 3 groups are in :mod:`bythebooktest.groups_l2` and :mod:`bythebooktest.groups_l3`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from . import render
from .kb import Claim, KBBuilder, Pred, allowed, can, does, membership, must, vp_pred
from .logic import Verdict, literal
from .plan import LEVEL_DEPTHS, ClaimSpec
from .tables import VP, AbilityRow, FactRow, Tables
from .vocab import Vocabulary

NEUTRAL_VERBS = [
    ("sing", "sings"),
    ("dance", "dances"),
    ("whistle", "whistles"),
    ("paint", "paints"),
    ("juggle", "juggles"),
    ("knit", "knits"),
]


class NotEnoughContent(RuntimeError):
    """The curated tables do not have enough rows for the claims requested."""


@dataclass
class Context:
    """What a group needs to build a claim: random generator, vocabulary, tables and the rulebook."""

    rng: random.Random
    vocab: Vocabulary
    tables: Tables
    kb: KBBuilder

    def take_fact_row(self) -> FactRow:
        """A row not yet used in this rulebook, from the category with the fewest claims so far."""
        available = [r for r in self.tables.facts if r.id not in self.kb.used_rows]
        if not available:
            raise NotEnoughContent(f"{self.kb.kb_id}: no fact rows left")
        counts = self.kb.category_counts
        least = min(counts.get(r.category, 0) for r in available)
        row = self.rng.choice([r for r in available if counts.get(r.category, 0) == least])
        self.kb.used_rows.add(row.id)
        return row


# --- relation with the real world -------------------------------------------------------------


def world_truth(kind: str, positive: bool) -> bool:
    """Real-world value of a literal: kind ``"true"`` or ``"trap"`` is true, ``"false"`` is false."""
    base = kind in ("true", "trap")
    return base if positive else not base


def required_truth(expected: Verdict, relation: str) -> bool:
    """Real-world value the claim must have to get the requested relation."""
    if expected is Verdict.TRUE:
        return relation == "agree"
    if expected is Verdict.FALSE:
        return relation == "conflict"
    return relation == "unsupported_true"


def relation_of(expected: Verdict, truth: bool) -> str:
    """Relation with the real world of a claim with this expected answer and real-world value."""
    if expected is Verdict.TRUE:
        return "agree" if truth else "conflict"
    if expected is Verdict.FALSE:
        return "conflict" if truth else "agree"
    return "unsupported_true" if truth else "unsupported_false"


def statement_world(kind: str, positive: bool) -> str:
    return "agree" if world_truth(kind, positive) else "conflict"


def row_literals(row: FactRow) -> list[tuple[VP, str]]:
    return [(row.true, "true")] + [(v, "false") for v in row.false]


def pick_literal(
    rng: random.Random, row: FactRow, spec: ClaimSpec, polarity: bool | None = None
) -> tuple[VP, str, bool]:
    want = required_truth(spec.expected, spec.relation)
    options = [
        (vp, kind, pos)
        for vp, kind in row_literals(row)
        for pos in (True, False)
        if (polarity is None or pos == polarity) and world_truth(kind, pos) == want
    ]
    if not options:
        raise NotEnoughContent(f"{row.id}: no literal for {spec}")
    return rng.choice(options)


def trap_literal(rng: random.Random, row: FactRow, exclude: str) -> tuple[VP, str]:
    """A statement close to the claim that does not decide it: a written condition or another value."""
    if row.traps:
        return rng.choice(row.traps), "trap"
    others = [(vp, kind) for vp, kind in row_literals(row) if vp.pos != exclude]
    return rng.choice(others)


class Group:
    """A kind of reasoning. Subclasses set ``name`` and implement :meth:`depths` and :meth:`build`."""

    name = ""

    def depths(self, level: int) -> list[int]:
        """Reasoning depths this group can build at ``level``."""
        raise NotImplementedError

    def variants(self, level: int) -> list[str]:
        """Proof schemes this group rotates through at ``level``."""
        return [""]

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        """Add the statements the claim needs to ``ctx.kb`` and return the claim."""
        raise NotImplementedError


def rule_depths(level: int) -> list[int]:
    """Allowed depths for groups built on rules: at least one rule."""
    lo, hi = LEVEL_DEPTHS[level]
    return list(range(max(lo, 1), hi + 1))


# --- claim target and supporting chains (used by the Level 2 and 3 groups) ----------------------


@dataclass
class Target:
    """The literal of the claim: subject, predicate and sign, with its relation with the real world."""

    subject: str
    person: bool
    category: str | None
    pred: Pred
    positive: bool
    relation: str


def claim_polarity(expected: Verdict, derived: bool | None) -> bool | None:
    """Sign the claim must have if the rulebook proves the predicate with sign ``derived``."""
    if derived is None or expected is Verdict.UNKNOWN:
        return None
    return derived if expected is Verdict.TRUE else not derived


def derived_polarity(expected: Verdict, claim_positive: bool, default: bool | None) -> bool:
    """Sign the rulebook must prove to give the claim its expected answer."""
    if expected is Verdict.TRUE:
        return claim_positive
    if expected is Verdict.FALSE:
        return not claim_positive
    return True if default is None else default


def choose_target(ctx: Context, spec: ClaimSpec, polarity: bool | None) -> Target:
    """Choose the claim literal: invented words for neutral claims, a table row otherwise."""
    if spec.relation == "neutral":
        positive = polarity if polarity is not None else ctx.rng.choice([True, False])
        return Target(ctx.vocab.name(), False, None, membership(ctx.vocab.noun()), positive, "neutral")
    row = ctx.take_fact_row()
    vp, kind, positive = pick_literal(ctx.rng, row, spec, polarity)
    return Target(
        row.subject,
        row.kind == "person",
        row.category,
        vp_pred(vp.pos, vp.neg),
        positive,
        relation_of(spec.expected, world_truth(kind, positive)),
    )


def derive(ctx: Context, t: Target, pred: Pred, positive: bool, k: int, owner: str) -> list[str]:
    """Statements that prove ``pred(subject)`` with the given sign using ``k`` simple rules.

    ``k = 0``: the fact is written. ``k >= 1``: "S is a c1. Every c1 is a c2. ... Every ck is (not)
    pred." The intermediate classes are invented, so the statements are neutral.
    """
    return derive_on(ctx, t.subject, t.person, t.category, pred, positive, k, owner)


def derive_on(
    ctx: Context, subject: str, person: bool, category: str | None, pred: Pred, positive: bool, k: int, owner: str
) -> list[str]:
    kb = ctx.kb
    common: dict[str, Any] = dict(role="proof", world="neutral", category=category, owner=owner)
    if k == 0:
        return [kb.fact(subject, pred, positive, **common)]
    chain = [membership(ctx.vocab.noun()) for _ in range(k)]
    sids = [kb.fact(subject, chain[0], True, **common)]
    for i in range(k - 1):
        sids.append(kb.rule(chain[i], chain[i + 1], True, person=person, **common))
    sids.append(kb.rule(chain[-1], pred, positive, person=person, **common))
    return sids


def ability_rows(ctx: Context, family: str) -> list[AbilityRow]:
    """Rows of abilities.toml that a family of groups may still use in this rulebook."""
    return [r for r in ctx.tables.abilities if ctx.kb.ability_family.get(r.id, family) == family]


def custom_claim(
    ctx: Context,
    cid: str,
    group: str,
    subtype: str,
    text: str,
    logic,
    spec: ClaimSpec,
    relation: str,
    category: str | None,
    reasoning: list[str],
    support: list[str],
) -> Claim:
    """A claim that is not a literal about one subject (sets, counts, "every"): Level 3."""
    if relation != spec.relation:
        raise AssertionError(f"built relation {relation} differs from the requested one {spec}")
    claim = Claim(
        id=cid,
        group=group,
        subtype=subtype,
        text=text,
        logic=logic,
        expected=spec.expected,
        world_relation=relation,
        category=category,
        depth=spec.depth,
        reasoning=[] if spec.expected is Verdict.UNKNOWN else reasoning,
        support=support,
    )
    ctx.kb.add_claim(claim)
    return claim


def finish(
    ctx: Context,
    cid: str,
    group: str,
    spec: ClaimSpec,
    t: Target,
    reasoning: list[str],
    support: list[str],
    traps: list[str] | None = None,
) -> Claim:
    if spec.expected is Verdict.UNKNOWN:
        reasoning = []
    return _claim(
        ctx,
        cid,
        group,
        spec.variant,
        t.subject,
        t.pred,
        t.positive,
        spec,
        t.relation,
        t.category,
        reasoning,
        support,
        traps,
    )


def _claim(
    ctx: Context,
    cid: str,
    group: str,
    subtype: str,
    subject: str,
    pred: Pred,
    positive: bool,
    spec: ClaimSpec,
    relation: str,
    category: str | None,
    reasoning: list[str],
    support: list[str],
    traps: list[str] | None = None,
) -> Claim:
    if relation != spec.relation:
        raise AssertionError(f"built relation {relation} differs from the requested one {spec}")
    claim = Claim(
        id=cid,
        group=group,
        subtype=subtype,
        text=render.fact(subject, pred.vp(positive)),
        logic=literal(pred.id, subject, positive),
        expected=spec.expected,
        world_relation=relation,
        category=category,
        depth=spec.depth,
        reasoning=reasoning,
        support=support,
        traps=traps or [],
    )
    ctx.kb.add_claim(claim)
    return claim


class DirectFact(Group):
    """The claim, or its negation, is written in the rulebook; for UNKNOWN, only a nearby trap is."""

    name = "direct_fact"

    def depths(self, level: int) -> list[int]:
        return [0]

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng = ctx.kb, ctx.rng
        cid = kb.new_claim_id()
        common: dict[str, Any] = dict(owner=cid)

        if spec.relation == "neutral":
            subject, category = ctx.vocab.name(), None
            pred, positive, kind = membership(ctx.vocab.noun()), rng.choice([True, False]), None
            relation = "neutral"
        else:
            row = ctx.take_fact_row()
            subject, category = row.subject, row.category
            vp, kind, positive = pick_literal(rng, row, spec)
            pred = vp_pred(vp.pos, vp.neg)
            relation = relation_of(spec.expected, world_truth(kind, positive))

        def world(pos: bool) -> str:
            return "neutral" if kind is None else statement_world(kind, pos)

        support: list[str] = []
        traps: list[str] = []
        if spec.expected is Verdict.TRUE:
            support.append(
                kb.fact(subject, pred, positive, role="proof", world=world(positive), category=category, **common)
            )
        elif spec.expected is Verdict.FALSE:
            support.append(
                kb.fact(
                    subject, pred, not positive, role="proof", world=world(not positive), category=category, **common
                )
            )
        else:
            if kind is None:
                other, other_world = membership(ctx.vocab.noun()), "neutral"
            else:
                tvp, tkind = trap_literal(rng, row, exclude=vp.pos)
                other, other_world = vp_pred(tvp.pos, tvp.neg), statement_world(tkind, True)
            traps.append(kb.fact(subject, other, True, role="trap", world=other_world, category=category, **common))

        reasoning = [] if spec.expected is Verdict.UNKNOWN else ["lookup"]
        return _claim(
            ctx, cid, self.name, "fact", subject, pred, positive, spec, relation, category, reasoning, support, traps
        )


class RuleChain(Group):
    """The subject belongs to a class; simple rules lead to the predicate of the claim.

    Forward chains have positive conclusions only (negated conclusions belong to the negation group),
    so a TRUE claim is positive and a FALSE claim is the negation of what the chain proves. For
    UNKNOWN one link of the chain is missing, but the rule that concludes the claim's predicate stays.
    """

    name = "rule_chain"

    def depths(self, level: int) -> list[int]:
        return rule_depths(level)

    def variants(self, level: int) -> list[str]:
        # Backward reasoning starts at Level 2.
        return ["forward"] if level == 1 else ["forward", "backward"]

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        if spec.variant == "backward":
            return self._backward(spec, ctx)
        kb, rng, vocab = ctx.kb, ctx.rng, ctx.vocab
        cid = kb.new_claim_id()
        d = spec.depth
        polarity = {Verdict.TRUE: True, Verdict.FALSE: False}.get(spec.expected)

        start_alt: tuple[VP, str] | None = None
        if spec.relation == "neutral":
            subject, category, person, row = vocab.name(), None, False, None
            target = membership(vocab.noun())
            positive = polarity if polarity is not None else rng.choice([True, False])
            relation = "neutral"
        else:
            row = ctx.take_fact_row()
            subject, category, person = row.subject, row.category, row.kind == "person"
            vp, kind, positive = pick_literal(rng, row, spec, polarity)
            target = vp_pred(vp.pos, vp.neg)
            relation = relation_of(spec.expected, world_truth(kind, positive))
            if d >= 2 and rng.random() < 0.5:
                alts = [(v, k) for v, k in row_literals(row) if v.pos != vp.pos]
                if alts:
                    start_alt = rng.choice(alts)

        chain: list[Pred] = [membership(vocab.noun()) for _ in range(d)] + [target]
        start_world = "neutral"
        if start_alt is not None:
            chain[0] = vp_pred(start_alt[0].pos, start_alt[0].neg)
            start_world = statement_world(start_alt[1], True)

        missing = rng.randrange(d) if spec.expected is Verdict.UNKNOWN else None
        common: dict[str, Any] = dict(owner=cid, category=category)
        support = [kb.fact(subject, chain[0], True, role="proof", world=start_world, **common)]
        for i in range(d):
            if i == missing:
                if i == d - 1:
                    # The rule that concludes the claim's predicate stays, from another class.
                    detached = membership(vocab.noun())
                    support.append(
                        kb.rule(detached, target, True, person=person, role="proof", world="neutral", **common)
                    )
                continue
            support.append(
                kb.rule(chain[i], chain[i + 1], True, person=person, role="proof", world="neutral", **common)
            )

        reasoning = [] if spec.expected is Verdict.UNKNOWN else ["modus_ponens"]
        return _claim(
            ctx, cid, self.name, "forward", subject, target, positive, spec, relation, category, reasoning, support
        )

    def _backward(self, spec: ClaimSpec, ctx: Context) -> Claim:
        """Backward (modus tollens): "Everyone who was born in Australia is a c1. Every c1 is a c2. S is
        not a c2.", so S was not born in Australia. For UNKNOWN the rulebook says that S is a c2:
        affirming the consequent does not decide the claim.
        """
        kb, vocab = ctx.kb, ctx.vocab
        cid = kb.new_claim_id()
        t = choose_target(ctx, spec, claim_polarity(spec.expected, False))
        chain = [t.pred] + [membership(vocab.noun()) for _ in range(spec.depth)]
        common: dict[str, Any] = dict(role="proof", world="neutral", category=t.category, owner=cid)
        support = [kb.rule(chain[i], chain[i + 1], True, person=t.person, **common) for i in range(spec.depth)]
        unknown = spec.expected is Verdict.UNKNOWN
        support.append(kb.fact(t.subject, chain[-1], unknown, **common))
        return finish(ctx, cid, self.name, spec, t, ["modus_tollens"], support)


class Modal(Group):
    """Modals: "generally", "can", "must", "is allowed to" (rules 7 and 8 of the prompt).

    TRUE and FALSE claims repeat the modal of the rule ("can" from "can"); UNKNOWN claims ask about a
    different property ("does" from "can", or from "generally"), which the rulebook does not settle.
    """

    name = "modal"

    def depths(self, level: int) -> list[int]:
        return [1]

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng, vocab = ctx.kb, ctx.rng, ctx.vocab
        cid = kb.new_claim_id()
        unknown = spec.expected is Verdict.UNKNOWN
        neutral = spec.relation == "neutral"

        if neutral:
            options = ["can_does", "must_can", "allowed_does"]
        else:
            options = ["can_does"]
        if unknown:
            options.append("generally")
        subtype = rng.choice(options)

        individual = vocab.name()
        want = None if neutral else required_truth(spec.expected, spec.relation)

        # Predicate and sign of the claim.
        if neutral:
            noun = vocab.noun()
            singular, plural, category = noun, render.plural_invented(noun), None
            verb, third = rng.choice(NEUTRAL_VERBS)
            ability: AbilityRow | None = None
        else:
            ability, claim_positive = self._pick_ability(ctx, subtype, spec, want)
            singular, plural, category = ability.singular, ability.plural, ability.category
            verb, third = ability.verb, ability.third

        rules: dict[str, Pred | None] = {
            "can_does": can(verb),
            "must_can": must(verb),
            "allowed_does": allowed(verb),
            "generally": None,
        }
        rule_pred = rules[subtype]
        claim_pred: Pred | None
        if unknown:
            claim_pred = {
                "can_does": does(verb, third),
                "generally": does(verb, third),
                "must_can": can(verb),
                "allowed_does": does(verb, third),
            }[subtype]
        else:
            claim_pred = rule_pred
        assert claim_pred is not None  # "generally" is used for UNKNOWN claims only
        if neutral:
            positive = {Verdict.TRUE: True, Verdict.FALSE: False}.get(spec.expected, rng.choice([True, False]))
            relation = "neutral"
        else:
            assert ability is not None
            positive = claim_positive
            truth = ability.world_does if claim_pred.id.startswith("does:") else ability.world_can
            relation = relation_of(spec.expected, truth if positive else not truth)

        common: dict[str, Any] = dict(owner=cid, category=category)
        support = [kb.fact(individual, membership(singular), True, role="proof", world="neutral", **common)]
        if subtype == "generally":
            world = "neutral" if ability is None else ("agree" if ability.world_does else "conflict")
            support.append(kb.generally(render.capitalize_first(plural), verb, role="proof", world=world, **common))
        else:
            assert rule_pred is not None
            if ability is None or subtype != "can_does":
                world = "neutral"
            else:
                world = "agree" if ability.world_can else "conflict"
            support.append(
                kb.rule(membership(singular), rule_pred, True, person=False, role="proof", world=world, **common)
            )

        reasoning = [] if unknown else ["modus_ponens"]
        return _claim(
            ctx, cid, self.name, subtype, individual, claim_pred, positive, spec, relation, category, reasoning, support
        )

    @staticmethod
    def _pick_ability(ctx: Context, subtype: str, spec: ClaimSpec, want: bool | None) -> tuple[AbilityRow, bool]:
        """Choose the class and the sign of the claim so that its relation with the world is the one
        requested.
        """
        options: list[tuple[AbilityRow, bool]] = []
        for row in ability_rows(ctx, "modal"):
            if spec.expected is Verdict.UNKNOWN:
                for pos in (True, False):
                    if (row.world_does if pos else not row.world_does) == want:
                        options.append((row, pos))
            else:
                pos = spec.expected is Verdict.TRUE  # TRUE: "can v"; FALSE: "cannot v"
                if (row.world_can if pos else not row.world_can) == want:
                    options.append((row, pos))
        if not options:
            raise NotEnoughContent(f"abilities.toml: no row for {spec}")
        row, pos = ctx.rng.choice(options)
        ctx.kb.ability_family[row.id] = "modal"
        return row, pos
