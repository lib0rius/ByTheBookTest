"""Reasoning groups that start at Level 3.

- ``relation``: rules about relations between two and three entities;
- ``exists``: "at least one" ("Some penguin that lives in Zorbak can fly.");
- ``counting``: closed lists and counts ("Exactly two penguins that live in Zorbak can fly.");
- ``universal``: claims with "every", also about an empty set;
- ``shared_chain``: chains that share steps between different claims.

Sets are always "the members of a class that live in an invented place": the place isolates every claim
from the others of the same rulebook. The claim states "can <verb>", which works in singular and plural.
The real-world relation of a set claim follows a convention of this generator: the real-world value of
"a member of the class can <verb>" (``world_can`` in abilities.toml).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from . import render
from .groups import (
    NEUTRAL_VERBS,
    Context,
    Group,
    NotEnoughContent,
    ability_rows,
    choose_target,
    claim_polarity,
    custom_claim,
    derive_on,
    derived_polarity,
    finish,
    relation_of,
    required_truth,
    rule_depths,
)
from .kb import LIVES, Claim, Pred, can, member_of, membership
from .logic import And, Atom, Eq, Exists, ForAll, Formula, Implies, Not, Verdict, literal
from .plan import ClaimSpec

REL_PHRASES = ["is linked to", "is close to", "is paired with"]


@dataclass
class SetContent:
    """What a set claim is about: the class, its plural, the property and the real-world relation."""

    cls: Pred
    plural: str
    prop: Pred
    category: str | None
    relation: str


def set_content(ctx: Context, spec: ClaimSpec) -> SetContent:
    """Choose class and property: invented for neutral claims, an ability row otherwise."""
    rng, vocab = ctx.rng, ctx.vocab
    if spec.relation == "neutral":
        noun = vocab.noun()
        verb, _ = rng.choice(NEUTRAL_VERBS)
        return SetContent(membership(noun), render.plural_invented(noun), can(verb), None, "neutral")
    want = required_truth(spec.expected, spec.relation)
    rows = [r for r in ability_rows(ctx, "sets") if r.world_can == want]
    if not rows:
        raise NotEnoughContent(f"abilities.toml: no free class for {spec}")
    row = rng.choice(rows)
    ctx.kb.ability_family[row.id] = "sets"
    return SetContent(
        membership(row.singular), row.plural, can(row.verb), row.category, relation_of(spec.expected, row.world_can)
    )


class _SetGroup(Group):
    """Base of the groups about the members of a class living in a place."""

    VARIANTS: ClassVar[tuple[str, ...]] = ()

    def depths(self, level: int) -> list[int]:
        return rule_depths(level)

    def variants(self, level: int) -> list[str]:
        return list(self.VARIANTS)

    def _setup(self, spec: ClaimSpec, ctx: Context):
        cid = ctx.kb.new_claim_id()
        c = set_content(ctx, spec)
        place = ctx.vocab.name()
        common: dict[str, Any] = dict(role="proof", world="neutral", category=c.category, owner=cid)
        return cid, c, place, common

    def _members(self, ctx: Context, c: SetContent, place: str, n: int, common: dict) -> tuple[list[str], list[str]]:
        names, sids = [], []
        for _ in range(n):
            m = ctx.vocab.name()
            names.append(m)
            sids.append(ctx.kb.fact(m, c.cls, True, **common))
            sids.append(ctx.kb.lives(m, place, **common))
        return names, sids

    def _props(
        self, ctx: Context, c: SetContent, positives: list[str], negatives: list[str], k: int, owner: str
    ) -> list[str]:
        """Properties of the members, proved with chains of ``k`` rules (one chain per sign)."""
        kb = ctx.kb
        common: dict[str, Any] = dict(role="proof", world="neutral", category=c.category, owner=owner)
        sids: list[str] = []
        for members, positive in ((positives, True), (negatives, False)):
            if not members:
                continue
            if k == 0:
                sids += [kb.fact(m, c.prop, positive, **common) for m in members]
                continue
            chain = [membership(ctx.vocab.noun()) for _ in range(k)]
            sids += [kb.fact(m, chain[0], True, **common) for m in members]
            for i in range(k - 1):
                sids.append(kb.rule(chain[i], chain[i + 1], True, person=False, **common))
            sids.append(kb.rule(chain[-1], c.prop, positive, person=False, **common))
        return sids

    def _scoped_chain(self, ctx: Context, c: SetContent, place: str, positive: bool, k: int, common: dict) -> list[str]:
        """A chain from the set to the property.

        For example: "Every penguin that lives in Zorbak is a c1. Every c1 is a c2. Every c2 can fly."
        """
        kb = ctx.kb
        if k == 0:
            return [kb.scoped_rule(c.cls, place, (c.prop, positive), **common)]
        chain = [membership(ctx.vocab.noun()) for _ in range(k)]
        sids = [kb.scoped_rule(c.cls, place, (chain[0], True), **common)]
        for i in range(k - 1):
            sids.append(kb.rule(chain[i], chain[i + 1], True, person=False, **common))
        sids.append(kb.rule(chain[-1], c.prop, positive, person=False, **common))
        return sids


class Exists_(_SetGroup):
    """Existence, "Some member can ...": one member proves it; a closed list can rule it out."""

    name = "exists"
    VARIANTS = ("direct", "closed")

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng = ctx.kb, ctx.rng
        cid, c, place, common = self._setup(spec, ctx)
        k = spec.depth - 1
        e = spec.expected
        support: list[str] = []
        if spec.variant == "direct":
            if e is Verdict.TRUE:
                names, s = self._members(ctx, c, place, 1, common)
                support = s + self._props(ctx, c, names, [], k, cid)
            elif e is Verdict.FALSE:
                support = self._scoped_chain(ctx, c, place, False, k, common)
                support += self._members(ctx, c, place, rng.randint(0, 1), common)[1]
            else:
                # One member that cannot: the set is not closed, so there may be others.
                names, s = self._members(ctx, c, place, 1, common)
                support = s + self._props(ctx, c, [], names, k, cid)
            reasoning = ["existential"] if e is Verdict.TRUE else ["modus_ponens"]
        else:
            names, s = self._members(ctx, c, place, rng.randint(1, 3), common)
            support = [*s, kb.closed_list(c.cls, c.plural, place, names, **common)]
            if e is Verdict.TRUE:
                support += self._props(ctx, c, names[:1], names[1:], k, cid)
            elif e is Verdict.FALSE:
                support += self._props(ctx, c, [], names, k, cid)
            else:
                support += self._props(ctx, c, [], names[1:], k, cid)  # nothing known about names[0]
            reasoning = ["closed_list"]
        text = render.set_claim("exists", 0, c.cls.noun, c.plural, place, c.prop.pos)
        logic = Exists(("?x",), And((literal(c.cls.id, "?x"), Atom(LIVES, ("?x", place)), literal(c.prop.id, "?x"))))
        return custom_claim(
            ctx, cid, self.name, spec.variant, text, logic, spec, c.relation, c.category, reasoning, support
        )


class Universal(_SetGroup):
    """Universal, "Every member can ...": a rule or a closed list proves it, a counterexample disproves it."""

    name = "universal"
    VARIANTS = ("direct", "closed")

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng = ctx.kb, ctx.rng
        cid, c, place, common = self._setup(spec, ctx)
        k = spec.depth - 1
        e = spec.expected
        subtype = spec.variant
        if spec.variant == "direct":
            if e is Verdict.TRUE:
                support = self._scoped_chain(ctx, c, place, True, k, common)
            else:
                # FALSE: a counterexample. UNKNOWN: a single example does not prove "every".
                names, s = self._members(ctx, c, place, 1, common)
                positive = e is Verdict.UNKNOWN
                support = s + self._props(ctx, c, names if positive else [], [] if positive else names, k, cid)
            reasoning = ["modus_ponens"] if e is Verdict.TRUE else ["counterexample"]
        else:
            vacuous = e is Verdict.TRUE and rng.random() < 0.25
            names, s = self._members(ctx, c, place, 0 if vacuous else rng.randint(1, 3), common)
            support = [*s, kb.closed_list(c.cls, c.plural, place, names, **common)]
            if vacuous:
                subtype = "vacuous"  # "every" over an empty set is true (rule 9 of the prompt)
            elif e is Verdict.TRUE:
                support += self._props(ctx, c, names, [], k, cid)
            elif e is Verdict.FALSE:
                support += self._props(ctx, c, names[1:], names[:1], k, cid)
            else:
                support += self._props(ctx, c, names[1:], [], k, cid)  # nothing known about names[0]
            reasoning = ["closed_list"]
        text = render.set_claim("universal", 0, c.cls.noun, c.plural, place, c.prop.pos)
        logic = ForAll(("?x",), Implies(member_of(c.cls, place), literal(c.prop.id, "?x")))
        return custom_claim(ctx, cid, self.name, subtype, text, logic, spec, c.relation, c.category, reasoning, support)


def at_least(c: SetContent, place: str, n: int) -> Formula:
    """At least ``n`` different members have the property."""
    xs = tuple(f"?x{i}" for i in range(n))
    parts: list[Formula] = []
    for x in xs:
        parts += [literal(c.cls.id, x), Atom(LIVES, (x, place)), literal(c.prop.id, x)]
    for i in range(n):
        for j in range(i + 1, n):
            parts.append(Not(Eq(xs[i], xs[j])))
    return Exists(xs, And(tuple(parts)))


def count_formula(c: SetContent, place: str, n: int, exact: bool) -> Formula:
    """Count: "at least n" or "exactly n" members have the property (different names, different entities).

    "Exactly n" is written "at least n and not at least n+1": equivalent to "there are n members and
    every other member is one of them", but without alternating "exists" and "for every", so Z3 always
    decides it.
    """
    if not exact:
        return at_least(c, place, n)
    return And((at_least(c, place, n), Not(at_least(c, place, n + 1))))


class Counting(_SetGroup):
    """Closed lists and counts: different names denote different entities, and a set is complete only
    if the rulebook says so (rules 3 and 4 of the prompt).
    """

    name = "counting"
    VARIANTS = ("exactly", "at_least")

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng = ctx.kb, ctx.rng
        cid, c, place, common = self._setup(spec, ctx)
        k = spec.depth - 1
        e = spec.expected
        n = rng.randint(1, 3)
        extra = rng.randint(0, 2)
        closed = True
        if spec.variant == "exactly":
            if e is Verdict.TRUE:
                pos, neg, unknown = n, extra, 0
            elif e is Verdict.FALSE:
                pos, neg, unknown = (n + 1, extra, 0) if rng.random() < 0.5 else (n - 1, extra + 1, 0)
            elif rng.random() < 0.5:
                pos, neg, unknown = n, extra, 1  # nothing known about one member
            else:
                pos, neg, unknown, closed = n, extra, 0, False  # the set is not closed
        else:
            if e is Verdict.TRUE:
                pos, neg, unknown, closed = n, extra, 0, rng.random() < 0.5
            elif e is Verdict.FALSE:
                pos, neg, unknown = n - 1, extra + 1, 0
            else:
                pos, neg, unknown, closed = n - 1, extra, 0, False
        names, support = self._members(ctx, c, place, pos + neg + unknown, common)
        if closed:
            support.append(kb.closed_list(c.cls, c.plural, place, names, **common))
        support += self._props(ctx, c, names[:pos], names[pos : pos + neg], k, cid)
        text = render.set_claim(spec.variant, n, c.cls.noun, c.plural, place, c.prop.pos)
        logic = count_formula(c, place, n, spec.variant == "exactly")
        return custom_claim(
            ctx, cid, self.name, spec.variant, text, logic, spec, c.relation, c.category, ["counting"], support
        )


class Relation(Group):
    """Rules about relations. Direction matters: "A is linked to B" does not say that B is linked to A,
    and that is the trap for UNKNOWN.
    """

    name = "relation"
    Q: ClassVar[dict[str, bool | None]] = {"two_entities": None, "three_entities": None, "backward": False}

    def depths(self, level: int) -> list[int]:
        return rule_depths(level)

    def variants(self, level: int) -> list[str]:
        return list(self.Q)

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng, vocab = ctx.kb, ctx.rng, ctx.vocab
        cid = kb.new_claim_id()
        q0 = self.Q[spec.variant]
        t = choose_target(ctx, spec, claim_polarity(spec.expected, q0))
        q = derived_polarity(spec.expected, t.positive, q0)
        unknown = spec.expected is Verdict.UNKNOWN
        common: dict[str, Any] = dict(role="proof", world="neutral", category=t.category, owner=cid)
        p1, p2 = rng.sample(REL_PHRASES, 2)
        cls = membership(vocab.noun())
        k = spec.depth - 1
        if spec.variant == "two_entities":
            o = vocab.name()
            support = [kb.rel_rule(p1, cls, (t.pred, q), person=t.person, **common)]
            if unknown and rng.random() < 0.5:
                support.append(kb.rel_fact(o, p1, t.subject, **common))  # opposite direction
                support += derive_on(ctx, o, False, t.category, cls, True, k, cid)
            else:
                support.append(kb.rel_fact(t.subject, p1, o, **common))
                if not unknown:
                    support += derive_on(ctx, o, False, t.category, cls, True, k, cid)
            reasoning = ["modus_ponens", "relation"]
        elif spec.variant == "three_entities":
            o1, o2 = vocab.name(), vocab.name()
            support = [
                kb.rel3_rule(p1, p2, cls, (t.pred, q), person=t.person, **common),
                kb.rel_fact(t.subject, p1, o1, **common),
            ]
            # UNKNOWN: either the second link goes the opposite way, or the class of o2 is missing.
            brk = rng.choice(["reverse", "no_class"]) if unknown else None
            if brk == "reverse":
                support.append(kb.rel_fact(o2, p2, o1, **common))
            else:
                support.append(kb.rel_fact(o1, p2, o2, **common))
            if brk != "no_class":
                support += derive_on(ctx, o2, False, t.category, cls, True, max(0, k - 1), cid)
            reasoning = ["modus_ponens", "relation"]
        else:
            # "Everyone who was born in Australia is linked to only kelvars. S is linked to O.
            # O is not a kelvar.", so S was not born in Australia. UNKNOWN: O is a kelvar.
            o = vocab.name()
            plural = render.plural_invented(cls.class_noun)
            support = [
                kb.only_rule(t.pred, p1, cls, plural, person=t.person, **common),
                kb.rel_fact(t.subject, p1, o, **common),
            ]
            support += derive_on(ctx, o, False, t.category, cls, unknown, k, cid)
            reasoning = ["modus_tollens", "relation"]
        return finish(ctx, cid, self.name, spec, t, reasoning, support)


class SharedChain(Group):
    """Chains that share steps: several claims of the same rulebook join a common chain of invented
    classes (a "hub") and leave it with a rule of their own. UNKNOWN: the subject joins after the exit
    point, and rules go one way only.
    """

    name = "shared_chain"
    HUB_LENGTH = 8

    def depths(self, level: int) -> list[int]:
        return rule_depths(level)

    def variants(self, level: int) -> list[str]:
        return ["shared"]

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, rng, vocab = ctx.kb, ctx.rng, ctx.vocab
        cid = kb.new_claim_id()
        t = choose_target(ctx, spec, None)
        q = derived_polarity(spec.expected, t.positive, None)
        common: dict[str, Any] = dict(role="proof", world="neutral", category=t.category, owner=cid)
        d = spec.depth
        # On a chain every predicate exits once only. Two exits with the same predicate, for example
        # "No femoven equals nine" and further on "Every buzastis equals nine", contradict each other as
        # soon as an entity joins the chain before both.
        free = [h for h, hub in enumerate(kb.hubs) if t.pred.id not in kb.hub_exits[h]]
        if free and rng.random() < 0.75:
            h = rng.choice(free)
        else:
            kb.hubs.append([membership(vocab.noun()) for _ in range(self.HUB_LENGTH)])
            kb.hub_exits.append(set())
            h = len(kb.hubs) - 1
        hub = kb.hubs[h]
        kb.hub_exits[h].add(t.pred.id)
        i = rng.randint(0, len(hub) - d - 1)
        j = i + d - 1
        attach = j + 1 if spec.expected is Verdict.UNKNOWN else i
        support = [kb.fact(t.subject, hub[attach], True, **common)]
        for idx in range(i, j):
            support.append(kb.rule(hub[idx], hub[idx + 1], True, person=False, **common))
        support.append(kb.rule(hub[j], t.pred, q, person=t.person, **common))
        return finish(ctx, cid, self.name, spec, t, ["modus_ponens", "shared_steps"], support)


L3_GROUPS = [Relation(), Exists_(), Counting(), Universal(), SharedChain()]
