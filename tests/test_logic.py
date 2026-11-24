"""Expected answers computed in classical logic."""

from bythebooktest.logic import (
    And,
    Atom,
    Cmp,
    Eq,
    ForAll,
    Implies,
    Not,
    Or,
    Verdict,
    Xor,
    evaluate,
    inconsistency_core,
    literal,
    rule,
)


def test_a_crashed_process_fails_only_its_own_job():
    # Z3 can kill its process outright: only that job must fail, not all of them.
    import operator
    import os

    from bythebooktest.logic import run_isolated

    jobs = [("crash", os._exit, (3,)), ("a", operator.add, (2, 3)), ("b", operator.mul, (4, 5))]
    got = dict(run_isolated(jobs, 2, on_crash=lambda key: f"{key} crashed"))
    assert got == {"crash": "crash crashed", "a": 5, "b": 20}


def verdicts(kb, claims):
    ev = evaluate(kb, claims)
    assert ev.consistent
    return ev.verdicts


def test_modus_ponens_true_false_unknown():
    kb = [literal("p", "a"), rule([("p", True)], ("q", True))]
    assert verdicts(kb, [literal("q", "a"), literal("q", "a", False), literal("r", "a")]) == [
        Verdict.TRUE,
        Verdict.FALSE,
        Verdict.UNKNOWN,
    ]


def test_modus_tollens_is_classical():
    kb = [rule([("p", True)], ("q", True)), literal("q", "a", False)]
    assert verdicts(kb, [literal("p", "a")]) == [Verdict.FALSE]


def test_missing_link_is_unknown():
    kb = [literal("p", "a"), rule([("r", True)], ("q", True))]
    assert verdicts(kb, [literal("q", "a"), literal("q", "a", False)]) == [
        Verdict.UNKNOWN,
        Verdict.UNKNOWN,
    ]


def test_exclusion_is_not_assumed():
    # Rule 2 of the prompt: "born in Australia" does not rule out "born in Corsica".
    kb = [literal("born_australia", "napoleon")]
    assert verdicts(kb, [literal("born_corsica", "napoleon")]) == [Verdict.UNKNOWN]


def test_xor_and_case_analysis():
    kb = [
        Xor(Atom("sailor", ("a",)), Atom("painter", ("a",))),
        literal("sailor", "a"),
        Or((Atom("p", ("b",)), Atom("q", ("b",)))),
        ForAll(("?x",), Implies(Atom("p", ("?x",)), Atom("r", ("?x",)))),
        ForAll(("?x",), Implies(Atom("q", ("?x",)), Atom("r", ("?x",)))),
    ]
    assert verdicts(kb, [literal("painter", "a"), literal("r", "b")]) == [
        Verdict.FALSE,
        Verdict.TRUE,
    ]


def test_inconsistent_kb_is_detected():
    ev = evaluate([literal("p", "a"), Not(Atom("p", ("a",)))], [literal("p", "a")])
    assert not ev.consistent


def test_inconsistency_core_finds_the_culprits():
    kb = [literal("q", "b"), literal("p", "a"), rule([("p", True)], ("r", False)), literal("s", "c"), literal("r", "a")]
    assert inconsistency_core(kb) == [1, 2, 4]


def _born_after(year_limit):
    y = "?y:int"
    return ForAll(
        ("?x", y), Implies(And((Atom("year:born", ("?x", y)), Cmp(">", y, year_limit))), Atom("young", ("?x",)))
    )


def test_years_and_exclusive_bounds():
    # Rule 10 of the prompt: "after 1970" excludes 1970.
    kb = [_born_after(1970), Atom("year:born", ("napoleon", 1975)), Atom("year:born", ("rex", 1970))]
    assert verdicts(kb, [literal("young", "napoleon"), literal("young", "rex"), literal("young", "kiro")]) == [
        Verdict.TRUE,
        Verdict.UNKNOWN,
        Verdict.UNKNOWN,
    ]


def test_interval_includes_both_years():
    y = "?y:int"
    rule_ = ForAll(
        ("?x", y),
        Implies(
            And((Atom("year:built", ("?x", y)), And((Cmp(">=", y, 1990), Cmp("<=", y, 2000))))), Atom("old", ("?x",))
        ),
    )
    kb = [rule_, Atom("year:built", ("a", 1990)), Atom("year:built", ("b", 2000)), Atom("year:built", ("c", 2001))]
    assert verdicts(kb, [literal("old", e) for e in ("a", "b", "c")]) == [
        Verdict.TRUE,
        Verdict.TRUE,
        Verdict.UNKNOWN,
    ]


def test_exception_pattern():
    kb = [
        ForAll(("?x",), Implies(And((Atom("bird", ("?x",)), Not(Atom("penguin", ("?x",))))), Atom("flies", ("?x",)))),
        rule([("penguin", True)], ("flies", False)),
        literal("bird", "tweety"),
        literal("bird", "pingu"),
        literal("penguin", "pingu"),
        literal("bird", "robin"),
        literal("penguin", "robin", False),
    ]
    assert verdicts(kb, [literal("flies", e) for e in ("tweety", "pingu", "robin")]) == [
        Verdict.UNKNOWN,
        Verdict.FALSE,
        Verdict.TRUE,
    ]


def test_parallel_evaluation_matches_serial():
    kb = [literal(f"p{i}", "a") for i in range(0, 40, 2)]
    kb += [rule([(f"p{i}", True)], (f"q{i}", True)) for i in range(40)]
    claims = [literal(f"q{i}", "a", i % 3 != 0) for i in range(40)]
    serial = evaluate(kb, claims).verdicts
    seen = []
    parallel = evaluate(kb, claims, workers=4, progress=lambda d, t: seen.append((d, t))).verdicts
    assert parallel == serial
    assert seen and seen[-1] == (40, 40)


def test_closed_lists_and_counting():
    from bythebooktest.groups_l3 import SetContent, count_formula
    from bythebooktest.kb import LIVES, can, member_of, membership

    c = SetContent(membership("penguin"), "penguins", can("fly"), None, "neutral")

    def member(m):
        return [literal(c.cls.id, m), Atom(LIVES, (m, "Zorbak"))]

    closed = ForAll(
        ("?y",), Implies(member_of(c.cls, "Zorbak", "?y"), Or((Eq("?y", "a"), Eq("?y", "b"), Eq("?y", "c"))))
    )
    props = [literal(c.prop.id, "a"), literal(c.prop.id, "b"), literal(c.prop.id, "c", False)]
    kb = member("a") + member("b") + member("c") + props
    claims = [
        count_formula(c, "Zorbak", 2, True),
        count_formula(c, "Zorbak", 2, False),
        count_formula(c, "Zorbak", 3, False),
    ]
    # Without a closed list "exactly two" is unknown; "at least two" holds (different names, rule 3).
    assert verdicts(kb, claims) == [Verdict.UNKNOWN, Verdict.TRUE, Verdict.UNKNOWN]
    assert verdicts([*kb, closed], claims) == [Verdict.TRUE, Verdict.TRUE, Verdict.FALSE]


def test_universal_on_empty_set_is_true():
    from bythebooktest.kb import member_of, membership

    nobody = ForAll(("?y",), Not(member_of(membership("penguin"), "Zorbak", "?y")))
    claim = ForAll(("?x",), Implies(member_of(membership("penguin"), "Zorbak"), Atom("can:fly", ("?x",))))
    assert verdicts([nobody], [claim]) == [Verdict.TRUE]


def test_generally_has_no_logical_content():
    # "Birds generally fly" has no logical content: the claim stays UNKNOWN (rule 7 of the prompt).
    kb = [literal("isa_bird", "tweety")]
    assert verdicts(kb, [literal("does_fly", "tweety")]) == [Verdict.UNKNOWN]
