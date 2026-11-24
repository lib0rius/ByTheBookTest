"""Building a rulebook: predicates, statements, claims and their labels.

:class:`KBBuilder` collects the statements and claims of one rulebook. Each statement is written twice,
as an English sentence (through :mod:`bythebooktest.render`) and as a formula (through
:mod:`bythebooktest.logic`), so that every expected answer can be checked with Z3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import render
from .logic import (
    And,
    Atom,
    Cmp,
    Eq,
    ForAll,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Verdict,
    Xor,
    literal,
    rule,
    to_text,
)

LIVES = "rel:lives in"


def member_of(cls: Pred, place: str, var: str = "?x") -> Formula:
    """Formula for "``var`` is a member of the class and lives in the place" (the sets of Level 3)."""
    return And((literal(cls.id, var), Atom(LIVES, (var, place))))


ROLE_PRIORITY = {"proof": 0, "trap": 1, "noise": 2}
"""When the same statement serves several roles, the lowest value wins."""


@dataclass(frozen=True)
class Pred:
    """A predicate about one entity, with its positive and negative phrase (third person singular)."""

    id: str
    pos: str
    neg: str
    noun: str | None = None
    """Set for class membership ("is a kelvar"), which is rendered differently in rules."""

    def vp(self, positive: bool) -> str:
        return self.pos if positive else self.neg

    @property
    def class_noun(self) -> str:
        """The class noun, for predicates that must be class memberships."""
        if self.noun is None:
            raise ValueError(f"{self.id} is not a class membership")
        return self.noun


def membership(noun: str) -> Pred:
    """Class membership: "is a <noun>"."""
    a_noun = render.with_article(noun)
    return Pred(id=f"isa:{noun}", pos=f"is {a_noun}", neg=f"is not {a_noun}", noun=noun)


def vp_pred(pos: str, neg: str) -> Pred:
    """A verb phrase from the fact tables, for example "was born in Corsica"."""
    return Pred(id=f"vp:{pos.lower()}", pos=pos, neg=neg)


def can(verb: str) -> Pred:
    return Pred(id=f"can:{verb}", pos=f"can {verb}", neg=f"cannot {verb}")


def does(verb: str, third: str) -> Pred:
    return Pred(id=f"does:{verb}", pos=third, neg=f"does not {verb}")


def must(verb: str) -> Pred:
    return Pred(id=f"must:{verb}", pos=f"must {verb}", neg=f"does not have to {verb}")


def allowed(verb: str) -> Pred:
    return Pred(id=f"allowed:{verb}", pos=f"is allowed to {verb}", neg=f"is not allowed to {verb}")


@dataclass
class Statement:
    """A statement of the rulebook under construction."""

    id: str
    form: str  # fact | rule | generally
    text: str
    logic: Formula | None  # None for statements with no logical content ("generally ...")
    role: str  # proof | trap | noise
    world: str  # conflict | agree | neutral
    category: str | None


@dataclass
class Claim:
    """A claim under construction, with what it needs from the rulebook."""

    id: str
    group: str
    subtype: str
    text: str
    logic: Formula
    expected: Verdict
    world_relation: str  # conflict | agree | unsupported_true | unsupported_false | neutral
    category: str | None
    depth: int
    reasoning: list[str]
    support: list[str]  # statements of the proof (or of the incomplete chain, for UNKNOWN)
    traps: list[str] = field(default_factory=list)
    z3: Verdict | None = None


class KBBuilder:
    """Collects the statements and claims of one rulebook; a statement equal to an existing one is not
    added twice, and every statement remembers which claims use it.
    """

    def __init__(self, kb_id: str, level: int) -> None:
        self.kb_id = kb_id
        self.level = level
        self.statements: dict[str, Statement] = {}
        self.claims: dict[str, Claim] = {}
        self._by_key: dict[str, str] = {}
        self._owners: dict[str, set[str]] = {}  # statement -> claims that use it
        self._next_statement = 1
        self._next_claim = 1
        self.used_rows: set[str] = set()
        self.category_counts: dict[str, int] = {}  # non-neutral claims per fact category
        self.years: dict[str, int] = {}  # year of every dated subject (group "time")
        # Rows of abilities.toml: "modal" writes rules about a whole class ("Every penguin can fly"), the
        # set groups only about members living in a place. A class stays with one of the two families.
        self.ability_family: dict[str, str] = {}
        self.hubs: list[list[Pred]] = []  # shared chains (group "shared_chain")
        self.hub_exits: list[set[str]] = []  # for every chain, the predicates already used as exits

    # --- statements --------------------------------------------------------------------------

    def _add(
        self,
        form: str,
        text: str,
        logic: Formula | None,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        key = to_text(logic) if logic is not None else f"text:{text}"
        sid = self._by_key.get(key)
        if sid is None:
            sid = f"{self.kb_id}-S{self._next_statement:04d}"
            self._next_statement += 1
            self.statements[sid] = Statement(sid, form, text, logic, role, world, category)
            self._by_key[key] = sid
            self._owners[sid] = set()
        else:
            st = self.statements[sid]
            if ROLE_PRIORITY[role] < ROLE_PRIORITY[st.role]:
                st.role = role
        if owner is not None:
            self._owners[sid].add(owner)
        return sid

    def fact(
        self,
        subject: str,
        pred: Pred,
        positive: bool,
        *,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        return self._add(
            "fact",
            render.fact(subject, pred.vp(positive)),
            literal(pred.id, subject, positive),
            role,
            world,
            category,
            owner,
        )

    def rule(
        self,
        premise: Pred,
        conclusion: Pred,
        positive: bool = True,
        *,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        if premise.noun is not None:
            text = (
                render.rule_class(premise.noun, conclusion.pos)
                if positive
                else render.rule_class_negative(premise.noun, conclusion.pos)
            )
        else:
            text = (
                render.rule_generic(premise.pos, conclusion.pos, person)
                if positive
                else render.rule_generic_negative(premise.pos, conclusion.pos, person)
            )
        logic = rule([(premise.id, True)], (conclusion.id, positive))
        return self._add("rule", text, logic, role, world, category, owner)

    def rule_multi(
        self,
        premises: list[tuple[Pred, bool]],
        conclusion: tuple[Pred, bool],
        connective: str = "and",
        *,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """A rule with negated conditions or several conditions joined by "and" or "or"."""
        lits = [literal(p.id, "?x", pos) for p, pos in premises]
        body = lits[0] if len(lits) == 1 else (And(tuple(lits)) if connective == "and" else Or(tuple(lits)))
        logic = ForAll(("?x",), Implies(body, literal(conclusion[0].id, "?x", conclusion[1])))
        first, first_pos = premises[0]
        first_noun = first.noun if first_pos else None
        text = render.rule_two_premises(
            first_noun, [p.vp(pos) for p, pos in premises], connective, conclusion[0].vp(conclusion[1]), person
        )
        return self._add("rule", text, logic, role, world, category, owner)

    def rule_compound(
        self,
        premise: Pred,
        c1: tuple[Pred, bool],
        c2: tuple[Pred, bool],
        op: str,
        *,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """A rule with an "or" or "either ... or" conclusion: "Every kelvar was born in Australia or is a
        trusap."
        """
        a, b = literal(c1[0].id, "?x", c1[1]), literal(c2[0].id, "?x", c2[1])
        concl = Or((a, b)) if op == "or" else Xor(a, b)
        logic = ForAll(("?x",), Implies(literal(premise.id, "?x"), concl))
        text = render.rule_compound(premise.noun, premise.pos, c1[0].vp(c1[1]), c2[0].vp(c2[1]), op, person)
        return self._add("rule", text, logic, role, world, category, owner)

    def fact_compound(
        self,
        subject: str,
        p1: tuple[Pred, bool],
        p2: tuple[Pred, bool],
        op: str,
        *,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        a, b = literal(p1[0].id, subject, p1[1]), literal(p2[0].id, subject, p2[1])
        logic = Or((a, b)) if op == "or" else Xor(a, b)
        text = render.fact_compound(subject, p1[0].vp(p1[1]), p2[0].vp(p2[1]), op)
        return self._add("fact", text, logic, role, world, category, owner)

    def iff(
        self, a: Pred, b: Pred, *, person: bool, role: str, world: str, category: str | None, owner: str | None
    ) -> str:
        logic = ForAll(("?x",), Iff(literal(a.id, "?x"), literal(b.id, "?x")))
        return self._add("rule", render.iff(a.pos, b.pos, person), logic, role, world, category, owner)

    def year_fact(
        self, subject: str, verb: str, year: int, *, role: str, world: str, category: str | None, owner: str | None
    ) -> str:
        logic = Atom(f"year:{verb}", (subject, year))
        return self._add("fact", render.year_fact(subject, verb, year), logic, role, world, category, owner)

    def year_rule(
        self,
        verb: str,
        op: str,
        a: int,
        b: int | None,
        conclusion: tuple[Pred, bool],
        *,
        scope: Pred | None = None,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """A rule about dates: "Every kelvar that was born after 1970 is young." (rule 10 of the prompt).

        ``scope`` limits the rule to the members of a class: without it, the rule would hold for every
        dated subject of the rulebook, and two different claims could contradict each other about a
        third subject.
        """
        y = "?y:int"
        cond: Formula
        if op == "after":
            cond = Cmp(">", y, a)
        elif op == "before":
            cond = Cmp("<", y, a)
        else:
            assert b is not None, "an interval needs two years"
            cond = And((Cmp(">=", y, a), Cmp("<=", y, b)))
        parts: list[Formula] = [Atom(f"year:{verb}", ("?x", y)), cond]
        if scope is not None:
            parts.insert(0, literal(scope.id, "?x"))
        pred, positive = conclusion
        logic = ForAll(("?x", y), Implies(And(tuple(parts)), literal(pred.id, "?x", positive)))
        text = render.year_rule(
            verb,
            op,
            a,
            b,
            pred.pos,
            person,
            negative_pos_vp=None if positive else pred.pos,
            scope_noun=scope.noun if scope is not None else None,
        )
        return self._add("rule", text, logic, role, world, category, owner)

    # --- Level 3: relations, places, closed lists ---------------------------------------------

    def rel_fact(
        self, subject: str, phrase: str, obj: str, *, role: str, world: str, category: str | None, owner: str | None
    ) -> str:
        logic = Atom(f"rel:{phrase}", (subject, obj))
        return self._add("fact", render.rel_fact(subject, phrase, obj), logic, role, world, category, owner)

    def rel_rule(
        self,
        phrase: str,
        cls: Pred,
        conclusion: tuple[Pred, bool],
        *,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """L(x, y) and P(y) -> Q(x)."""
        pred, positive = conclusion
        body = And((Atom(f"rel:{phrase}", ("?x", "?y")), literal(cls.id, "?y")))
        logic = ForAll(("?x", "?y"), Implies(body, literal(pred.id, "?x", positive)))
        text = render.rel_rule(phrase, cls.class_noun, pred.pos, person, negative_pos_vp=None if positive else pred.pos)
        return self._add("rule", text, logic, role, world, category, owner)

    def rel3_rule(
        self,
        phrase1: str,
        phrase2: str,
        cls: Pred,
        conclusion: tuple[Pred, bool],
        *,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """L(x, y) and M(y, z) and P(z) -> Q(x): a rule over three entities."""
        pred, positive = conclusion
        body = And((Atom(f"rel:{phrase1}", ("?x", "?y")), Atom(f"rel:{phrase2}", ("?y", "?z")), literal(cls.id, "?z")))
        logic = ForAll(("?x", "?y", "?z"), Implies(body, literal(pred.id, "?x", positive)))
        text = render.rel3_rule(
            phrase1, phrase2, cls.class_noun, pred.pos, person, negative_pos_vp=None if positive else pred.pos
        )
        return self._add("rule", text, logic, role, world, category, owner)

    def only_rule(
        self,
        premise: Pred,
        phrase: str,
        cls: Pred,
        plural: str,
        *,
        person: bool,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """P(x) -> for every y, L(x, y) -> C(y), as in "Everyone who P is linked to only kelvars"."""
        inner = ForAll(("?y",), Implies(Atom(f"rel:{phrase}", ("?x", "?y")), literal(cls.id, "?y")))
        logic = ForAll(("?x",), Implies(literal(premise.id, "?x"), inner))
        text = render.only_rule(premise.pos, phrase, plural, person)
        return self._add("rule", text, logic, role, world, category, owner)

    def lives(self, subject: str, place: str, *, role: str, world: str, category: str | None, owner: str | None) -> str:
        return self._add(
            "fact", render.lives(subject, place), Atom(LIVES, (subject, place)), role, world, category, owner
        )

    def scoped_rule(
        self,
        cls: Pred,
        place: str,
        conclusion: tuple[Pred, bool],
        *,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """A rule about the members of a class in a place: "Every penguin that lives in Zorbak can fly."."""
        pred, positive = conclusion
        logic = ForAll(("?x",), Implies(member_of(cls, place), literal(pred.id, "?x", positive)))
        text = render.scoped_rule(cls.class_noun, place, pred.pos, negative=not positive)
        return self._add("rule", text, logic, role, world, category, owner)

    def closed_list(
        self,
        cls: Pred,
        plural: str,
        place: str,
        members: list[str],
        *,
        role: str,
        world: str,
        category: str | None,
        owner: str | None,
    ) -> str:
        """A closed list: the only members of a class living in a place (rule 4 of the prompt)."""
        if not members:
            logic = ForAll(("?y",), Not(member_of(cls, place, "?y")))
        else:
            eqs = [Eq("?y", m) for m in members]
            logic = ForAll(("?y",), Implies(member_of(cls, place, "?y"), eqs[0] if len(eqs) == 1 else Or(tuple(eqs))))
        text = render.closed_list(cls.class_noun, plural, place, members)
        return self._add("fact", text, logic, role, world, category, owner)

    def remove_noise(self, statement_id: str) -> None:
        st = self.statements.pop(statement_id)
        del self._owners[statement_id]
        del self._by_key[to_text(st.logic) if st.logic is not None else f"text:{st.text}"]

    def text_only(self, text: str, *, role: str, world: str, category: str | None, owner: str | None) -> str:
        """A statement with no useful logical content, for example "Napoleon was born." (no date)."""
        return self._add("fact", render.sentence(text), None, role, world, category, owner)

    def generally(
        self, plural: str, verb: str, *, role: str, world: str, category: str | None, owner: str | None
    ) -> str:
        return self._add("generally", render.generally(plural, verb), None, role, world, category, owner)

    # --- claims --------------------------------------------------------------------------------

    def new_claim_id(self) -> str:
        cid = f"{self.kb_id}-C{self._next_claim:04d}"
        self._next_claim += 1
        return cid

    def add_claim(self, claim: Claim) -> None:
        self.claims[claim.id] = claim
        if claim.category is not None:
            self.category_counts[claim.category] = self.category_counts.get(claim.category, 0) + 1

    def remove_claim(self, claim_id: str) -> None:
        """Remove a claim and the statements that only it used (noise stays)."""
        claim = self.claims.pop(claim_id)
        if claim.category is not None:
            self.category_counts[claim.category] -= 1
        for sid in list(self._owners):
            owners = self._owners[sid]
            owners.discard(claim_id)
            if not owners and self.statements[sid].role != "noise":
                st = self.statements.pop(sid)
                del self._owners[sid]
                key = to_text(st.logic) if st.logic is not None else f"text:{st.text}"
                del self._by_key[key]

    def logic_statements(self) -> list[Formula]:
        return [s.logic for s in self.statements.values() if s.logic is not None]

    def logic_statement_ids(self) -> list[str]:
        return [s.id for s in self.statements.values() if s.logic is not None]

    def owners(self, statement_id: str) -> set[str]:
        return set(self._owners.get(statement_id, set()))
