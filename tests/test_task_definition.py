"""Symbolic checks of the task definition: the three answers and the rules of interpretation.

Every example is decided twice, with two independent encodings that both use Z3:

- ``pipeline``: the formulas of :mod:`bythebooktest.logic` and its :func:`evaluate`, the code that labelled
  the dataset (open domain, different names for different entities);
- ``raw``: the same rulebook written directly with the Z3 Python API, without the project code.

The general propositions of the definition are checked by Z3 as first-order validities, with the models of a
rulebook represented as the elements of a sort: ``G(m)`` says that model ``m`` satisfies the rulebook,
``F(m)`` that it satisfies the claim.
"""

from __future__ import annotations

import pytest
import z3

from bythebooktest.logic import (
    And,
    Atom,
    Cmp,
    Eq,
    Exists,
    ForAll,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Verdict,
    Xor,
    evaluate,
)

T, F, U = Verdict.TRUE, Verdict.FALSE, Verdict.UNKNOWN


# --- The raw encoding ------------------------------------------------------------------------------------------

Ent = z3.DeclareSort("Ent")


def ent(name: str) -> z3.ExprRef:
    return z3.Const(name, Ent)


def pred(name: str, arity: int = 1, int_last: bool = False) -> z3.FuncDeclRef:
    sorts = [Ent] * arity
    if int_last:
        sorts[-1] = z3.IntSort()
    return z3.Function(name, *sorts, z3.BoolSort())


TIMEOUT_MS = 60_000
"""Time limit of every check: an undecided check fails the test instead of hanging."""


def check_with(s: z3.Solver, extra: z3.BoolRef) -> z3.CheckSatResult:
    s.push()
    s.add(extra)
    result = s.check()
    s.pop()
    return result


def raw_label(gamma: list[z3.BoolRef], phi: z3.BoolRef, names: list[z3.ExprRef], una: bool = True) -> Verdict:
    """TRUE if gamma ∪ {¬phi} is unsatisfiable, FALSE if gamma ∪ {phi} is, UNKNOWN if both are satisfiable."""
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(*gamma)
    if una and len(names) > 1:
        s.add(z3.Distinct(*names))
    assert s.check() == z3.sat, "the rulebook must be consistent"
    without_claim = check_with(s, z3.Not(phi))
    with_claim = check_with(s, phi)
    assert z3.unknown not in (without_claim, with_claim)
    if without_claim == z3.unsat:
        return T
    if with_claim == z3.unsat:
        return F
    return U


def pipe_label(kb: list[Formula], claim: Formula) -> Verdict:
    ev = evaluate(kb, [claim])
    assert ev.consistent
    assert ev.verdicts[0] is not None
    return ev.verdicts[0]


def a(p: str, *args) -> Atom:
    return Atom(p, tuple(args))


x, y, z_ = z3.Consts("x y z", Ent)
n = z3.Int("n")


# --- Examples of the definition, each with both encodings -----------------------------------------------------


def test_forward_inference():
    """Every sailor owns a boat; Napoleon is a sailor; so Napoleon owns a boat."""
    sailor, boat = pred("sailor"), pred("owns_boat")
    nap = ent("napoleon")
    gamma = [z3.ForAll([x], z3.Implies(sailor(x), boat(x))), sailor(nap)]
    assert raw_label(gamma, boat(nap), [nap]) == T
    kb = [ForAll(("?x",), Implies(a("sailor", "?x"), a("owns_boat", "?x"))), a("sailor", "Napoleon")]
    assert pipe_label(kb, a("owns_boat", "Napoleon")) == T


def test_backward_inference():
    """Rule 1 of the prompt: every Tarv is a Pelm, Kiro is not a Pelm, so Kiro is not a Tarv."""
    tarv, pelm = pred("tarv"), pred("pelm")
    kiro = ent("kiro")
    gamma = [z3.ForAll([x], z3.Implies(tarv(x), pelm(x))), z3.Not(pelm(kiro))]
    assert raw_label(gamma, tarv(kiro), [kiro]) == F
    kb = [ForAll(("?x",), Implies(a("tarv", "?x"), a("pelm", "?x"))), Not(a("pelm", "Kiro"))]
    assert pipe_label(kb, a("tarv", "Kiro")) == F


def test_exclusions_only_if_written():
    """Rule 2: "Kiro lives in Dask" does not imply "Kiro does not live in Morv"."""
    lives = pred("lives_in", 2)
    kiro, dask, morv = ent("kiro"), ent("dask"), ent("morv")
    assert raw_label([lives(kiro, dask)], z3.Not(lives(kiro, morv)), [kiro, dask, morv]) == U
    assert pipe_label([a("lives_in", "Kiro", "Dask")], Not(a("lives_in", "Kiro", "Morv"))) == U


def _at_least_two(moon, zeta):
    return z3.Exists([x, y], z3.And(x != y, moon(x, zeta), moon(y, zeta)))


def _at_least_three(moon, zeta):
    return z3.Exists([x, y, z_], z3.And(z3.Distinct(x, y, z_), moon(x, zeta), moon(y, zeta), moon(z_, zeta)))


def _exactly_two(moon, zeta):
    """The generator's form of "exactly two": at least two and not at least three (no ∃∀ alternation)."""
    return z3.And(_at_least_two(moon, zeta), z3.Not(_at_least_three(moon, zeta)))


def _exactly_two_textbook(moon, zeta):
    """The textbook form: two different members, and every member is one of them."""
    only = z3.ForAll([z_], z3.Implies(moon(z_, zeta), z3.Or(z_ == x, z_ == y)))
    return z3.Exists([x, y], z3.And(x != y, moon(x, zeta), moon(y, zeta), only))


def _p_at_least(k: int) -> Formula:
    vs = tuple(f"?v{i}" for i in range(k))
    diff = [Not(Eq(vs[i], vs[j])) for i in range(k) for j in range(i + 1, k)]
    return Exists(vs, And((*diff, *(a("moon_of", v, "Zeta") for v in vs))))


def _p_at_least_two() -> Formula:
    return _p_at_least(2)


def _p_exactly_two() -> Formula:
    return And((_p_at_least(2), Not(_p_at_least(3))))


def test_two_forms_of_exactly_two_agree():
    """The generator's form of "exactly two" is equivalent to the textbook form: on every domain of 1 to 6
    entities (finite sorts), and on an unrestricted domain.
    """
    for size in range(1, 7):
        dom, elems = z3.EnumSort(f"D{size}", [f"d{size}_{i}" for i in range(size)])
        moon = z3.Function(f"moon{size}", dom, dom, z3.BoolSort())
        u, v, w, t = z3.Consts(f"u{size} v{size} w{size} t{size}", dom)
        zeta = elems[0]
        least2 = z3.Exists([u, v], z3.And(u != v, moon(u, zeta), moon(v, zeta)))
        least3 = z3.Exists([u, v, w], z3.And(z3.Distinct(u, v, w), moon(u, zeta), moon(v, zeta), moon(w, zeta)))
        only = z3.ForAll([t], z3.Implies(moon(t, zeta), z3.Or(t == u, t == v)))
        textbook = z3.Exists([u, v], z3.And(u != v, moon(u, zeta), moon(v, zeta), only))
        s = z3.Solver()
        s.set("timeout", TIMEOUT_MS)
        s.add(z3.Not(z3.And(least2, z3.Not(least3)) == textbook))
        assert s.check() == z3.unsat, f"forms differ on a domain of {size}"
    moon, zeta = pred("moon_of", 2), ent("zeta")
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(z3.Not(_exactly_two(moon, zeta) == _exactly_two_textbook(moon, zeta)))
    assert s.check() == z3.unsat  # equivalent on every domain


def test_open_domain_and_unique_names():
    """Io and Rhea are moons of Zeta: at least two moons (different names), not exactly two (open domain)."""
    moon = pred("moon_of", 2)
    io, rhea, zeta = ent("io"), ent("rhea"), ent("zeta")
    names = [io, rhea, zeta]
    gamma = [moon(io, zeta), moon(rhea, zeta)]
    assert raw_label(gamma, _at_least_two(moon, zeta), names) == T
    assert raw_label(gamma, _exactly_two(moon, zeta), names) == U
    # Without unique names, Io and Rhea could be the same moon.
    assert raw_label(gamma, _at_least_two(moon, zeta), names, una=False) == U
    kb = [a("moon_of", "Io", "Zeta"), a("moon_of", "Rhea", "Zeta")]
    assert pipe_label(kb, _p_at_least_two()) == T
    assert pipe_label(kb, _p_exactly_two()) == U


def test_closed_list():
    """Rule 4: "The only moons of Zeta are Io and Rhea" makes the group complete."""
    moon = pred("moon_of", 2)
    io, rhea, zeta = ent("io"), ent("rhea"), ent("zeta")
    closed = z3.ForAll([x], z3.Implies(moon(x, zeta), z3.Or(x == io, x == rhea)))
    gamma = [moon(io, zeta), moon(rhea, zeta), closed]
    assert raw_label(gamma, _exactly_two(moon, zeta), [io, rhea, zeta]) == T
    p_closed = ForAll(("?y",), Implies(a("moon_of", "?y", "Zeta"), Or((Eq("?y", "Io"), Eq("?y", "Rhea")))))
    kb = [a("moon_of", "Io", "Zeta"), a("moon_of", "Rhea", "Zeta"), p_closed]
    assert pipe_label(kb, _p_exactly_two()) == T


def test_inclusive_and_exclusive_or():
    """Rule 5: from "sailor or painter" and "sailor" nothing follows on painter; with "but not both" it does."""
    sailor, painter = pred("sailor"), pred("painter")
    nap = ent("napoleon")
    assert raw_label([z3.Or(sailor(nap), painter(nap)), sailor(nap)], painter(nap), [nap]) == U
    assert raw_label([z3.Xor(sailor(nap), painter(nap)), sailor(nap)], painter(nap), [nap]) == F
    s, p = a("sailor", "Napoleon"), a("painter", "Napoleon")
    assert pipe_label([Or((s, p)), s], p) == U
    assert pipe_label([Xor(s, p), s], p) == F


def test_if_and_only_if_both_ways():
    """Rule 6: "someone is a citizen if and only if they were born in Australia" works in both directions."""
    citizen, born = pred("citizen"), pred("born_au")
    k = ent("kiro")
    rule = z3.ForAll([x], citizen(x) == born(x))
    assert raw_label([rule, born(k)], citizen(k), [k]) == T
    assert raw_label([rule, z3.Not(citizen(k))], born(k), [k]) == F
    p_rule = ForAll(("?x",), Iff(a("citizen", "?x"), a("born_au", "?x")))
    assert pipe_label([p_rule, a("born_au", "Kiro")], a("citizen", "Kiro")) == T
    assert pipe_label([p_rule, Not(a("citizen", "Kiro"))], a("born_au", "Kiro")) == F


def test_exception_written_as_a_condition():
    """Every bird that is not a penguin flies; no penguin flies; Tweety is a bird: "Tweety flies" is UNKNOWN."""
    bird, penguin, flies = pred("bird"), pred("penguin"), pred("flies")
    tw = ent("tweety")
    gamma = [
        z3.ForAll([x], z3.Implies(z3.And(bird(x), z3.Not(penguin(x))), flies(x))),
        z3.ForAll([x], z3.Implies(penguin(x), z3.Not(flies(x)))),
        bird(tw),
    ]
    assert raw_label(gamma, flies(tw), [tw]) == U
    # Monotonic: adding that Tweety is not a penguin settles it.
    assert raw_label([*gamma, z3.Not(penguin(tw))], flies(tw), [tw]) == T
    kb = [
        ForAll(("?x",), Implies(And((a("bird", "?x"), Not(a("penguin", "?x")))), a("flies", "?x"))),
        ForAll(("?x",), Implies(a("penguin", "?x"), Not(a("flies", "?x")))),
        a("bird", "Tweety"),
    ]
    assert pipe_label(kb, a("flies", "Tweety")) == U


def test_generally_carries_no_formula():
    """Rule 7: "Birds generally fly" adds no formula, so "Tweety flies" stays UNKNOWN."""
    bird, flies = pred("bird"), pred("flies")
    tw = ent("tweety")
    assert raw_label([bird(tw)], flies(tw), [tw]) == U
    assert pipe_label([a("bird", "Tweety")], a("flies", "Tweety")) == U


def test_can_and_does_are_distinct():
    """Rule 8: every plane can fly, Gikrustok is a plane: "Gikrustok can fly" TRUE, "Gikrustok flies" UNKNOWN."""
    plane, can_fly, flies = pred("plane"), pred("can_fly"), pred("flies")
    g = ent("gikrustok")
    gamma = [z3.ForAll([x], z3.Implies(plane(x), can_fly(x))), plane(g)]
    assert raw_label(gamma, can_fly(g), [g]) == T
    assert raw_label(gamma, flies(g), [g]) == U
    kb = [ForAll(("?x",), Implies(a("plane", "?x"), a("can_fly", "?x"))), a("plane", "Gikrustok")]
    assert pipe_label(kb, a("can_fly", "Gikrustok")) == T
    assert pipe_label(kb, a("flies", "Gikrustok")) == U


def test_must_does_not_give_allowed():
    """Rule 8 again: "Kiro must vote" leaves "Kiro is allowed to vote" UNKNOWN (weaker than SDL)."""
    must, allowed = pred("must_vote"), pred("allowed_to_vote")
    k = ent("kiro")
    assert raw_label([must(k)], allowed(k), [k]) == U
    assert pipe_label([a("must_vote", "Kiro")], a("allowed_to_vote", "Kiro")) == U


def test_sdl_makes_obligation_imply_permission():
    """In Standard Deontic Logic, with the usual Kripke semantics and a serial accessibility relation,
    O p -> P p is valid; without seriality it is not. This is what the convention of the task gives up.
    """
    W = z3.DeclareSort("W")
    r = z3.Function("R", W, W, z3.BoolSort())
    p = z3.Function("p", W, z3.BoolSort())
    w, v = z3.Consts("w v", W)

    def obligatory(at):
        return z3.ForAll([v], z3.Implies(r(at, v), p(v)))

    def permitted(at):
        return z3.Exists([v], z3.And(r(at, v), p(v)))

    serial = z3.ForAll([w], z3.Exists([v], r(w, v)))
    claim = z3.ForAll([w], z3.Implies(obligatory(w), permitted(w)))
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(serial, z3.Not(claim))
    assert s.check() == z3.unsat  # valid on serial frames
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(z3.Not(claim))
    assert s.check() == z3.sat  # not valid without seriality


# --- Reading "can" as "does" and "must" as "does" ---------------------------------------------------------------
# A Kripke model: worlds 0 and 1, an accessibility relation and the worlds where p holds. Possibility at w: some
# accessible world has p; necessity (and obligation) at w: every accessible world has p.

TWO_WORLDS_S5 = ({(0, 0), (0, 1), (1, 0), (1, 1)}, {1})
"""Each of two worlds sees both: an equivalence relation, a frame of S5 (hence also of T and S4); p only at 1."""
TWO_WORLDS_SERIAL = ({(0, 1), (1, 1)}, {1})
"""World 0 sees 1, world 1 sees itself: a serial frame, as in Standard Deontic Logic; p only at 1."""


def _possible(model, w: int) -> bool:
    access, true_at = model
    return any((w, v) in access and v in true_at for v in (0, 1))


def _necessary(model, w: int) -> bool:
    access, true_at = model
    return all(v in true_at for v in (0, 1) if (w, v) in access)


def test_possibility_does_not_give_actuality_direct():
    """Direct evaluation: at world 0, "possibly p" holds and p does not, on an S5 frame; "obligatory p" holds and
    p does not, on a serial frame. So neither "can -> does" nor "must -> does" is valid there.
    """
    access, true_at = TWO_WORLDS_S5
    assert all((w, w) in access for w in (0, 1))  # reflexive
    assert all((v, w) in access for w, v in access)  # symmetric
    assert all((w, u) in access for w, v in access for x, u in access if v == x)  # transitive
    assert _possible(TWO_WORLDS_S5, 0) and 0 not in true_at
    access, true_at = TWO_WORLDS_SERIAL
    assert all(any((w, v) in access for v in (0, 1)) for w in (0, 1))  # serial
    assert _necessary(TWO_WORLDS_SERIAL, 0) and 0 not in true_at


def test_possibility_does_not_give_actuality_z3():
    """Z3: on frames that are reflexive, symmetric and transitive (S5), <>p -> p has a countermodel; on serial
    frames, O p -> p has one. Z3 finds both models, with any number of worlds.
    """
    W = z3.DeclareSort("W")
    r = z3.Function("R", W, W, z3.BoolSort())
    p = z3.Function("p", W, z3.BoolSort())
    w, v, u = z3.Consts("w v u", W)
    w0 = z3.Const("w0", W)
    s5 = [
        z3.ForAll([w], r(w, w)),
        z3.ForAll([w, v], z3.Implies(r(w, v), r(v, w))),
        z3.ForAll([w, v, u], z3.Implies(z3.And(r(w, v), r(v, u)), r(w, u))),
    ]
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(*s5, z3.Exists([v], z3.And(r(w0, v), p(v))), z3.Not(p(w0)))
    assert s.check() == z3.sat
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    serial = z3.ForAll([w], z3.Exists([v], r(w, v)))
    s.add(serial, z3.ForAll([v], z3.Implies(r(w0, v), p(v))), z3.Not(p(w0)))
    assert s.check() == z3.sat


def test_vacuous_truth_needs_the_rulebook_to_exclude():
    """Rule 9: "every moon of Zeta is red" is TRUE if the rulebook says there are no moons of Zeta, and UNKNOWN
    if it only names none (open domain).
    """
    moon, red = pred("moon_of", 2), pred("red")
    zeta = ent("zeta")
    claim = z3.ForAll([x], z3.Implies(moon(x, zeta), red(x)))
    assert raw_label([z3.ForAll([x], z3.Not(moon(x, zeta)))], claim, [zeta]) == T
    assert raw_label([z3.BoolVal(True)], claim, [zeta]) == U
    p_claim = ForAll(("?x",), Implies(a("moon_of", "?x", "Zeta"), a("red", "?x")))
    assert pipe_label([ForAll(("?x",), Not(a("moon_of", "?x", "Zeta")))], p_claim) == T


def test_every_and_some():
    """A counterexample makes "every" FALSE; one example makes "some" TRUE; one example does not make "every" TRUE."""
    moon, red = pred("moon"), pred("red")
    io = ent("io")
    every = z3.ForAll([x], z3.Implies(moon(x), red(x)))
    some = z3.Exists([x], z3.And(moon(x), red(x)))
    assert raw_label([moon(io), z3.Not(red(io))], every, [io]) == F
    assert raw_label([moon(io), red(io)], some, [io]) == T
    assert raw_label([moon(io), red(io)], every, [io]) == U
    p_every = ForAll(("?x",), Implies(a("moon", "?x"), a("red", "?x")))
    p_some = Exists(("?x",), And((a("moon", "?x"), a("red", "?x"))))
    assert pipe_label([a("moon", "Io"), Not(a("red", "Io"))], p_every) == F
    assert pipe_label([a("moon", "Io"), a("red", "Io")], p_some) == T
    assert pipe_label([a("moon", "Io"), a("red", "Io")], p_every) == U


@pytest.mark.parametrize(
    ("op", "year", "expected"),
    [("after", 1975, T), ("after", 1970, U), ("before", 1965, U), ("interval", 2025, T), ("interval", 2026, U)],
)
def test_years(op, year, expected):
    """Rule 10: "after" and "before" exclude the year, "from ... to ..." includes both ends."""
    born, young = pred("born_in", 2, int_last=True), pred("young")
    nap = ent("napoleon")
    if op == "after":
        cond, p_cond = n > 1970, Cmp(">", "?y:int", 1970)
    elif op == "before":
        cond, p_cond = n < 1965, Cmp("<", "?y:int", 1965)
    else:
        cond, p_cond = z3.And(n >= 2020, n <= 2025), And((Cmp(">=", "?y:int", 2020), Cmp("<=", "?y:int", 2025)))
    rule = z3.ForAll([x, n], z3.Implies(z3.And(born(x, n), cond), young(x)))
    assert raw_label([rule, born(nap, year)], young(nap), [nap]) == expected
    p_rule = ForAll(("?x", "?y:int"), Implies(And((a("born_in", "?x", "?y:int"), p_cond)), a("young", "?x")))
    assert pipe_label([p_rule, a("born_in", "Napoleon", year)], a("young", "Napoleon")) == expected


def test_an_inconsistent_rulebook_would_give_both_answers():
    """Why consistency is required: on an inconsistent rulebook both refutation checks succeed."""
    p = pred("p")
    k = ent("k")
    s = z3.Solver()
    s.add(p(k), z3.Not(p(k)))
    assert check_with(s, z3.Not(p(k))) == z3.unsat and check_with(s, p(k)) == z3.unsat
    assert evaluate([a("p", "K"), Not(a("p", "K"))], [a("p", "K")]).consistent is False


# --- General propositions, as first-order validities over models -----------------------------------------------

M = z3.DeclareSort("Model")
G = z3.Function("G", M, z3.BoolSort())  # m satisfies the rulebook
G1 = z3.Function("G1", M, z3.BoolSort())  # m satisfies a sub-rulebook
H = z3.Function("H", M, z3.BoolSort())  # m satisfies the removed statements
Fm = z3.Function("Fm", M, z3.BoolSort())  # m satisfies the claim
m = z3.Const("m", M)


def valid(f: z3.BoolRef, *assumptions: z3.BoolRef) -> bool:
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(*assumptions, z3.Not(f))
    result = s.check()
    assert result != z3.unknown, "Z3 did not decide"
    return result == z3.unsat


def entails(g, f):
    return z3.ForAll([m], z3.Implies(g(m), f(m)))


def not_f(at):
    return z3.Not(Fm(at))


def test_labels_are_exclusive_on_a_consistent_rulebook():
    """If Γ ⊨ φ and Γ ⊨ ¬φ, then Γ has no model."""
    assert valid(z3.Implies(z3.And(entails(G, Fm), entails(G, not_f)), z3.ForAll([m], z3.Not(G(m)))))


def test_labels_by_refutation():
    """Γ ⊨ φ exactly when Γ ∪ {¬φ} has no model."""
    assert valid(entails(G, Fm) == z3.Not(z3.Exists([m], z3.And(G(m), z3.Not(Fm(m))))))


def test_monotonicity_and_removing_statements():
    """With Γ = Γ' ∪ Δ: what Γ' entails, Γ entails; if Γ is consistent so is Γ'; if φ is UNKNOWN on Γ it is
    UNKNOWN on Γ' too (removing statements never turns UNKNOWN into TRUE or FALSE).
    """
    split = z3.ForAll([m], G(m) == z3.And(G1(m), H(m)))
    assert valid(z3.Implies(entails(G1, Fm), entails(G, Fm)), split)
    assert valid(z3.Implies(z3.Exists([m], G(m)), z3.Exists([m], G1(m))), split)
    unknown_on_g = z3.And(z3.Not(entails(G, Fm)), z3.Not(entails(G, not_f)))
    unknown_on_g1 = z3.And(z3.Not(entails(G1, Fm)), z3.Not(entails(G1, not_f)))
    assert valid(z3.Implies(unknown_on_g, unknown_on_g1), split)
    # The converse fails: removing a statement can turn TRUE into UNKNOWN.
    assert not valid(z3.Implies(entails(G, Fm), entails(G1, Fm)), split)
