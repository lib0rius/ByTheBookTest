"""Reasoning groups that start at Level 2: negation, conjunction, disjunction, iff, exception, time.

Every group has several variants (proof schemes). The claim is always a literal about a subject of the
tables (or an invented one, for the neutral part); intermediate classes are invented. The depth is the
number of rules applied: a supporting chain of (depth - 1) simple rules leads to the fact the scheme
starts from (see :func:`~bythebooktest.groups.derive`).

For every variant, ``Q`` says which sign of the claim's predicate the rulebook proves (``None``: either).
For UNKNOWN, the scheme leaves in the rulebook a statement that "almost" decides the claim: the typical
trap of the form (denying the antecedent, inclusive "or", an exception not ruled out, the bound of an
interval).
"""

from __future__ import annotations

from typing import Any, ClassVar

from .groups import (
    Context,
    Group,
    Target,
    choose_target,
    claim_polarity,
    derive,
    derived_polarity,
    finish,
    rule_depths,
)
from .kb import Claim, membership
from .logic import Verdict
from .plan import ClaimSpec

Schemes = dict[str, bool | None]
"""Variant name -> sign of the claim's predicate the rulebook proves (``None``: either)."""


class _Schemas(Group):
    """Base of the Level 2 groups: one variant per key of ``Q``, built on a supporting chain."""

    Q: ClassVar[Schemes] = {}

    def depths(self, level: int) -> list[int]:
        return rule_depths(level)

    def variants(self, level: int) -> list[str]:
        return list(self.Q)

    def _start(self, spec: ClaimSpec, ctx: Context) -> tuple[str, Target, bool, int, bool, dict[str, Any]]:
        """Common start: claim id, target, sign to prove, length of the supporting chain, whether the
        claim is UNKNOWN, and the provenance of the statements.
        """
        cid = ctx.kb.new_claim_id()
        q0 = self.Q[spec.variant]
        t = choose_target(ctx, spec, claim_polarity(spec.expected, q0))
        q = derived_polarity(spec.expected, t.positive, q0)
        common: dict[str, Any] = dict(role="proof", world="neutral", category=t.category, owner=cid)
        return cid, t, q, spec.depth - 1, spec.expected is Verdict.UNKNOWN, common


class Negation(_Schemas):
    """A negated conclusion or condition."""

    name = "negation"
    Q: ClassVar[Schemes] = {"neg_conclusion": False, "neg_condition": True, "backward": False}

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, vocab = ctx.kb, ctx.vocab
        cid, t, _, k, unknown, common = self._start(spec, ctx)
        a = membership(vocab.noun())
        if spec.variant == "neg_conclusion":
            # "S is a kelvar. No kelvar was born in Australia." UNKNOWN: S is not a kelvar.
            support = derive(ctx, t, a, not unknown, k, cid)
            support.append(kb.rule(a, t.pred, False, person=t.person, **common))
            reasoning = ["modus_ponens"]
        elif spec.variant == "neg_condition":
            # "S is not a kelvar. Everyone who is not a kelvar was born in Australia."
            support = derive(ctx, t, a, unknown, k, cid)
            support.append(kb.rule_multi([(a, False)], (t.pred, True), person=t.person, **common))
            reasoning = ["modus_ponens"]
        else:
            # "No one who was born in Australia is a kelvar. S is a kelvar." UNKNOWN: S is not a kelvar.
            support = derive(ctx, t, a, not unknown, k, cid)
            support.append(kb.rule(t.pred, a, False, person=t.person, **common))
            reasoning = ["modus_tollens"]
        return finish(ctx, cid, self.name, spec, t, reasoning, support)


class Conjunction(_Schemas):
    """A rule with several conditions joined by "and"."""

    name = "conjunction"
    Q: ClassVar[Schemes] = {"forward": None, "backward": False}

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, vocab, rng = ctx.kb, ctx.vocab, ctx.rng
        cid, t, q, k, unknown, common = self._start(spec, ctx)
        a, b = membership(vocab.noun()), membership(vocab.noun())
        if spec.variant == "forward":
            # "Every kelvar that is a trusap was born in Australia." UNKNOWN: a condition is missing.
            support = derive(ctx, t, a, True, k, cid)
            if not unknown:
                support.append(kb.fact(t.subject, b, True, **common))
            elif rng.random() < 0.5:
                support.append(kb.fact(t.subject, b, False, **common))
            support.append(kb.rule_multi([(a, True), (b, True)], (t.pred, q), person=t.person, **common))
            reasoning = ["modus_ponens"]
        else:
            # "Every kelvar that was born in Australia is a trusap. S is a kelvar. S is not a trusap."
            support = [] if unknown else derive(ctx, t, a, True, k, cid)
            support.append(kb.fact(t.subject, b, False, **common))
            support.append(kb.rule_multi([(a, True), (t.pred, True)], (b, True), person=t.person, **common))
            reasoning = ["modus_tollens"]
        return finish(ctx, cid, self.name, spec, t, reasoning, support)


class Disjunction(_Schemas):
    """Disjunction: "or" and "either ... or ..., but not both" in facts and rules (rule 5 of the prompt)."""

    name = "disjunction"
    Q: ClassVar[Schemes] = {
        "or_fact": True,
        "xor_fact": False,
        "or_condition": None,
        "or_conclusion": True,
        "xor_conclusion": False,
        "case_analysis": None,
    }

    def variants(self, level: int) -> list[str]:
        # Reasoning by cases starts at Level 3.
        return [v for v in self.Q if v != "case_analysis" or level >= 3]

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, vocab, rng = ctx.kb, ctx.vocab, ctx.rng
        cid, t, q, k, unknown, common = self._start(spec, ctx)
        a, b = membership(vocab.noun()), membership(vocab.noun())
        v = spec.variant
        if v == "case_analysis":
            # "S is a kelvar or a trusap. Every kelvar was born in Australia. Every trusap is a c1.
            # Every c1 was born in Australia." Both cases lead to the same result.
            # UNKNOWN: the rulebook covers one case only.
            support = [
                kb.fact_compound(t.subject, (a, True), (b, True), "or", **common),
                kb.rule(a, t.pred, q, person=t.person, **common),
            ]
            if not unknown:
                chain = [b] + [membership(vocab.noun()) for _ in range(max(0, k - 1))]
                for i in range(len(chain) - 1):
                    support.append(kb.rule(chain[i], chain[i + 1], True, person=t.person, **common))
                support.append(kb.rule(chain[-1], t.pred, q, person=t.person, **common))
            return finish(ctx, cid, self.name, spec, t, ["case_analysis"], support)
        if v in ("or_fact", "xor_fact"):
            pair = [(a, True), (t.pred, True)]
            rng.shuffle(pair)
            op = "or" if v == "or_fact" else "xor"
            support = [kb.fact_compound(t.subject, pair[0], pair[1], op, **common)]
            if v == "or_fact":
                # "S is a kelvar or was born in Australia. S is not a kelvar." UNKNOWN: S is a kelvar.
                support += derive(ctx, t, a, unknown, k, cid)
            elif not unknown:
                # "S either is a kelvar or was born in Australia, but not both. S is a kelvar."
                support += derive(ctx, t, a, True, k, cid)
            else:
                support += derive(ctx, t, b, True, k, cid)  # nothing about a: the xor does not decide
            reasoning = ["disjunctive_syllogism"]
        elif v == "or_condition":
            # "Everything that is a kelvar or is a trusap was born in Australia. S is a trusap."
            # UNKNOWN: S is not a kelvar, and the rulebook says nothing about trusap.
            support = derive(ctx, t, a, False, k, cid) if unknown else derive(ctx, t, b, True, k, cid)
            support.append(kb.rule_multi([(a, True), (b, True)], (t.pred, q), "or", person=t.person, **common))
            reasoning = ["modus_ponens"]
        else:
            # "Every kelvar was born in Australia or is a trusap. S is a kelvar. S is not a trusap."
            # xor: "... either ... or ..., but not both. S is a kelvar. S is a trusap."
            op = "or" if v == "or_conclusion" else "xor"
            support = derive(ctx, t, a, True, k, cid)
            if not unknown:
                support.append(kb.fact(t.subject, b, op == "xor", **common))
            support.append(kb.rule_compound(a, (t.pred, True), (b, True), op, person=t.person, **common))
            reasoning = ["modus_ponens", "disjunctive_syllogism"]
        return finish(ctx, cid, self.name, spec, t, reasoning, support)


class Biconditional(_Schemas):
    """Biconditional: "if and only if" (rule 6 of the prompt)."""

    name = "iff"
    Q: ClassVar[Schemes] = {"iff_forward": True, "iff_reverse": False}

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, vocab = ctx.kb, ctx.vocab
        cid, t, _, k, unknown, common = self._start(spec, ctx)
        a = membership(vocab.noun())
        support = [kb.iff(a, t.pred, person=t.person, **common)]
        if unknown:
            support += derive(ctx, t, membership(vocab.noun()), True, k, cid)  # nothing about a
        else:
            # forward: "S is a kelvar." reverse: "S is not a kelvar." (it works both ways)
            support += derive(ctx, t, a, spec.variant == "iff_forward", k, cid)
        return finish(ctx, cid, self.name, spec, t, ["biconditional"], support)


class Exception_(_Schemas):
    """Exceptions: "Every kelvar that is not a trusap was born in Australia. No trusap was born in
    Australia." UNKNOWN: the rulebook does not say whether S is a trusap.
    """

    name = "exception"
    Q: ClassVar[Schemes] = {"normal": True, "exception": False}

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, vocab = ctx.kb, ctx.vocab
        cid, t, _, k, unknown, common = self._start(spec, ctx)
        a, e = membership(vocab.noun()), membership(vocab.noun())
        support = [
            kb.rule_multi([(a, True), (e, False)], (t.pred, True), person=t.person, **common),
            kb.rule(e, t.pred, False, person=t.person, **common),
        ]
        support += derive(ctx, t, a, True, k, cid)
        if not unknown:
            support.append(kb.fact(t.subject, e, spec.variant == "exception", **common))
        return finish(ctx, cid, self.name, spec, t, ["modus_ponens"], support)


class Time(_Schemas):
    """Dates, validity intervals, missing dates (rule 10 of the prompt).

    UNKNOWN: the year equals the bound of "after" or "before" (excluded), the year is just outside an
    interval, or the date is not written.
    """

    name = "time"
    Q: ClassVar[Schemes] = {"after": None, "before": None, "interval": None}

    def build(self, spec: ClaimSpec, ctx: Context) -> Claim:
        kb, vocab, rng = ctx.kb, ctx.vocab, ctx.rng
        cid, t, q, k, unknown, common = self._start(spec, ctx)
        proper_person = t.person and t.subject[:1].isupper()
        verb = "was born" if proper_person else "was registered"
        # A subject has one year in the whole rulebook: two different years would look like a contradiction.
        dated = t.subject in kb.years
        year = kb.years.setdefault(t.subject, rng.randint(1950, 2020))
        v = spec.variant
        b = None
        if v == "after":
            a = year if unknown else year - rng.randint(1, 30)
        elif v == "before":
            a = year if unknown else year + rng.randint(1, 30)
        elif unknown:
            b = year - 1
            a = b - rng.randint(1, 10)
        else:
            a, b = year - rng.randint(0, 10), year + rng.randint(0, 10)

        support: list[str] = []
        if unknown and not dated and rng.random() < 0.5:
            support.append(kb.text_only(f"{t.subject} {verb}", **common))  # missing date
        else:
            # Invented years: for a real person born before 1950 the fact is in conflict with the world.
            world = "conflict" if proper_person else "neutral"
            support.append(
                kb.year_fact(t.subject, verb, year, role="proof", world=world, category=t.category, owner=cid)
            )
        # The date rule holds only for an invented class of the subject (see KBBuilder.year_rule).
        scope = membership(vocab.noun())
        support.append(kb.fact(t.subject, scope, True, **common))
        if k == 0:
            support.append(kb.year_rule(verb, v, a, b, (t.pred, q), scope=scope, person=t.person, **common))
        else:
            chain = [membership(vocab.noun()) for _ in range(k)]
            support.append(kb.year_rule(verb, v, a, b, (chain[0], True), scope=scope, person=t.person, **common))
            for i in range(k - 1):
                support.append(kb.rule(chain[i], chain[i + 1], True, person=t.person, **common))
            support.append(kb.rule(chain[-1], t.pred, q, person=t.person, **common))
        return finish(ctx, cid, self.name, spec, t, ["modus_ponens", "date_comparison"], support)


L2_GROUPS = [Negation(), Conjunction(), Disjunction(), Biconditional(), Exception_(), Time()]
