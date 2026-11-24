"""Logical forms of statements and claims, and the expected answer computed with Z3.

The expected answer of a claim follows classical first-order logic over the rulebook:

- TRUE if the rulebook entails the claim, FALSE if it entails its negation, UNKNOWN otherwise;
- a rulebook must be consistent (it never entails both);
- different names denote different entities, and the domain is open: there may be entities that
  no statement mentions.

Conventions:

- an argument starting with ``?`` is a variable; any other string argument is the name of an entity;
- a variable ending with ``:int`` and every integer argument are numbers (years);
- predicates are identified by a string, for example ``"isa:kelvar"`` or ``"vp:was born in australia"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import z3

Term = str | int


@dataclass(frozen=True)
class Atom:
    """A predicate applied to its arguments: ``Atom("isa:kelvar", ("Kiro",))``."""

    pred: str
    args: tuple[Term, ...]


@dataclass(frozen=True)
class Cmp:
    """A comparison between numbers; ``op`` is one of ``<``, ``<=``, ``>``, ``>=``, ``==``."""

    op: str
    left: Term
    right: Term


@dataclass(frozen=True)
class Eq:
    """Equality between entities (closed lists and counts)."""

    left: Term
    right: Term


@dataclass(frozen=True)
class Not:
    """Negation."""

    f: Formula


@dataclass(frozen=True)
class And:
    """Conjunction of any number of formulas."""

    fs: tuple[Formula, ...]


@dataclass(frozen=True)
class Or:
    """Inclusive disjunction of any number of formulas."""

    fs: tuple[Formula, ...]


@dataclass(frozen=True)
class Xor:
    """Exclusive disjunction: exactly one of the two."""

    a: Formula
    b: Formula


@dataclass(frozen=True)
class Implies:
    """Implication ``a -> b``."""

    a: Formula
    b: Formula


@dataclass(frozen=True)
class Iff:
    """Biconditional ``a <-> b``."""

    a: Formula
    b: Formula


@dataclass(frozen=True)
class ForAll:
    """Universal quantification over the given variables."""

    vars: tuple[str, ...]
    body: Formula


@dataclass(frozen=True)
class Exists:
    """Existential quantification over the given variables."""

    vars: tuple[str, ...]
    body: Formula


Formula = Atom | Cmp | Eq | Not | And | Or | Xor | Implies | Iff | ForAll | Exists


def is_var(name: Term) -> bool:
    return isinstance(name, str) and name.startswith("?")


def is_int_term(term: Term) -> bool:
    return isinstance(term, int) or (isinstance(term, str) and term.endswith(":int"))


def literal(pred: str, entity: str, positive: bool = True) -> Formula:
    atom = Atom(pred, (entity,))
    return atom if positive else Not(atom)


def rule(premises: list[tuple[str, bool]], conclusion: tuple[str, bool]) -> Formula:
    """A universal rule over one variable: premises joined by "and", a literal as conclusion."""
    prem = [literal(p, "?x", pos) for p, pos in premises]
    body_a = prem[0] if len(prem) == 1 else And(tuple(prem))
    return ForAll(("?x",), Implies(body_a, literal(conclusion[0], "?x", conclusion[1])))


def to_text(f: Formula) -> str:
    """A readable, stable text of the formula, used in the dataset files and to spot duplicates."""
    match f:
        case Atom(pred, args):
            return f"{pred}({', '.join(str(a) for a in args)})"
        case Cmp(op, left, right):
            return f"{left} {op} {right}"
        case Eq(left, right):
            return f"{left} = {right}"
        case Not(g):
            return f"not {_wrap(g)}"
        case And(fs):
            return " and ".join(_wrap(g) for g in fs)
        case Or(fs):
            return " or ".join(_wrap(g) for g in fs)
        case Xor(a, b):
            return f"{_wrap(a)} xor {_wrap(b)}"
        case Implies(a, b):
            return f"{_wrap(a)} -> {_wrap(b)}"
        case Iff(a, b):
            return f"{_wrap(a)} <-> {_wrap(b)}"
        case ForAll(vs, body):
            return f"forall {' '.join(vs)}. {to_text(body)}"
        case Exists(vs, body):
            return f"exists {' '.join(vs)}. {to_text(body)}"
    raise TypeError(f"unknown formula: {f!r}")


def _wrap(f: Formula) -> str:
    return to_text(f) if isinstance(f, (Atom, Not, Cmp, Eq)) else f"({to_text(f)})"


class Verdict(StrEnum):
    """The expected answer of a claim."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class _Encoder:
    """Translates formulas to Z3, with an open domain of entities."""

    sort: z3.SortRef = field(default_factory=lambda: z3.DeclareSort("Entity"))
    consts: dict[str, z3.ExprRef] = field(default_factory=dict)
    funcs: dict[tuple[str, tuple[bool, ...]], z3.FuncDeclRef] = field(default_factory=dict)
    _bound: int = 0

    def const(self, name: str) -> z3.ExprRef:
        if name not in self.consts:
            self.consts[name] = z3.Const(f"e{len(self.consts)}", self.sort)
        return self.consts[name]

    def func(self, pred: str, int_args: tuple[bool, ...]) -> z3.FuncDeclRef:
        key = (pred, int_args)
        if key not in self.funcs:
            sorts = [z3.IntSort() if is_int else self.sort for is_int in int_args]
            self.funcs[key] = z3.Function(f"p{len(self.funcs)}", *sorts, z3.BoolSort())
        return self.funcs[key]

    def term(self, t: Term, env: dict[str, z3.ExprRef]) -> z3.ExprRef:
        if isinstance(t, int):
            return z3.IntVal(t)
        if is_var(t):
            if t not in env:
                raise ValueError(f"free variable {t}")
            return env[t]
        return self.const(t)

    def encode(self, f: Formula, env: dict[str, z3.ExprRef] | None = None) -> z3.BoolRef:
        env = env or {}
        match f:
            case Atom(pred, args):
                terms = [self.term(a, env) for a in args]
                return self.func(pred, tuple(is_int_term(a) for a in args))(*terms)
            case Cmp(op, left, right):
                a, b = self.term(left, env), self.term(right, env)
                return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b}[op]
            case Eq(left, right):
                return self.term(left, env) == self.term(right, env)
            case Not(g):
                return z3.Not(self.encode(g, env))
            case And(fs):
                return z3.And([self.encode(g, env) for g in fs])
            case Or(fs):
                return z3.Or([self.encode(g, env) for g in fs])
            case Xor(a, b):
                return z3.Xor(self.encode(a, env), self.encode(b, env))
            case Implies(a, b):
                return z3.Implies(self.encode(a, env), self.encode(b, env))
            case Iff(a, b):
                return self.encode(a, env) == self.encode(b, env)
            case ForAll(vs, body) | Exists(vs, body):
                bound = {}
                for v in vs:
                    self._bound += 1
                    sort = z3.IntSort() if is_int_term(v) else self.sort
                    bound[v] = z3.Const(f"v{self._bound}", sort)
                inner = self.encode(body, {**env, **bound})
                quant = z3.ForAll if isinstance(f, ForAll) else z3.Exists
                return quant(list(bound.values()), inner)
        raise TypeError(f"unknown formula: {f!r}")

    def distinct(self) -> z3.BoolRef:
        """Different names denote different entities."""
        consts = list(self.consts.values())
        return z3.Distinct(*consts) if len(consts) > 1 else z3.BoolVal(True)


@dataclass
class Evaluation:
    """Result of :func:`evaluate`; a ``None`` verdict means Z3 did not decide in time."""

    consistent: bool
    verdicts: list[Verdict | None]


def _solver(kb: list[Formula], claims: list[Formula], timeout_ms: int):
    enc = _Encoder()
    kb_z3 = [enc.encode(f) for f in kb]
    claims_z3 = [enc.encode(c) for c in claims]  # registers the entities of the claims before Distinct
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for f in kb_z3:
        solver.add(f)
    solver.add(enc.distinct())
    return solver, claims_z3


def _check(solver: z3.Solver, extra: z3.BoolRef):
    """One check; an internal Z3 error (for example, out of memory) counts as undecided."""
    solver.push()
    try:
        solver.add(extra)
        return solver.check()
    except z3.Z3Exception:
        return z3.unknown
    finally:
        solver.pop()


def _verdict(solver: z3.Solver, c: z3.BoolRef) -> Verdict | None:
    without_claim = _check(solver, z3.Not(c))
    if without_claim == z3.unsat:
        return Verdict.TRUE
    with_claim = _check(solver, c)
    if with_claim == z3.unsat:
        return Verdict.FALSE
    if without_claim == z3.sat and with_claim == z3.sat:
        return Verdict.UNKNOWN
    return None


# State of each worker process: the rulebook is translated to Z3 once per process.
_WORKER: dict = {}


def _worker_init(kb: list[Formula], all_claims: list[Formula], timeout_ms: int) -> None:
    solver, claims_z3 = _solver(kb, all_claims, timeout_ms)
    _WORKER.update(solver=solver, claims=claims_z3)


def _worker_run(indices: list[int]) -> list[tuple[int, Verdict | None]]:
    solver, claims = _WORKER["solver"], _WORKER["claims"]
    return [(i, _verdict(solver, claims[i])) for i in indices]


def _atoms(f: Formula):
    match f:
        case Atom():
            yield f
        case Not(h):
            yield from _atoms(h)
        case And(fs) | Or(fs):
            for h in fs:
                yield from _atoms(h)
        case Xor(a, b) | Implies(a, b) | Iff(a, b):
            yield from _atoms(a)
            yield from _atoms(b)
        case ForAll(_, body) | Exists(_, body):
            yield from _atoms(body)


def fixed_positions(formulas: list[Formula]) -> dict[str, tuple[int, ...]]:
    """For every predicate, the argument positions that always hold an entity name, never a variable.

    For example, "lives in" always has a named place in second position ("... lives in Zorbak"). Then
    "lives in Zorbak" and "lives in Kelvorax" can be treated as distinct predicates: an equivalent
    renaming, because no statement talks about "lives in" with an arbitrary place.
    """
    variable: dict[str, set[int]] = {}
    arity: dict[str, int] = {}
    for f in formulas:
        for atom in _atoms(f):
            key = f"{atom.pred}/{len(atom.args)}"
            arity[key] = len(atom.args)
            variable.setdefault(key, set()).update(
                i for i, a in enumerate(atom.args) if not isinstance(a, str) or is_var(a)
            )
    return {k: tuple(i for i in range(n) if i not in variable[k]) for k, n in arity.items()}


def symbols(f: Formula, fixed: dict[str, tuple[int, ...]] | None = None) -> set[tuple[str, str]]:
    """Predicates and entity names that appear in a formula.

    With ``fixed``, a predicate is split by the values of its always-named positions.
    """
    out: set[tuple[str, str]] = set()

    def walk(g: Formula) -> None:
        match g:
            case Atom(pred, args):
                key = f"{pred}/{len(args)}"
                pos = (fixed or {}).get(key, ())
                out.add(("P", key + "".join(f"|{args[i]}" for i in pos)))
                out.update(("C", a) for a in args if isinstance(a, str) and not is_var(a))
            case Eq(left, right) | Cmp(_, left, right):
                out.update(("C", a) for a in (left, right) if isinstance(a, str) and not is_var(a))
            case Not(h):
                walk(h)
            case And(fs) | Or(fs):
                for h in fs:
                    walk(h)
            case Xor(a, b) | Implies(a, b) | Iff(a, b):
                walk(a)
                walk(b)
            case ForAll(_, body) | Exists(_, body):
                walk(body)

    walk(f)
    return out


def components(kb: list[Formula], claims: list[Formula]) -> list[tuple[list[int], list[int]]]:
    """Parts of the rulebook that share no symbol with each other, with the claims about each part.

    Two statements are in the same part if they share a predicate or an entity, directly or through
    other statements. A claim goes with the part of its symbols.
    """
    parent: dict[object, object] = {}

    def find(x: object) -> object:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: object, b: object) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    fixed = fixed_positions(kb + claims)
    for i, f in enumerate(kb):
        find(("S", i))
        for s in symbols(f, fixed):
            union(("S", i), s)
    for j, c in enumerate(claims):
        find(("Q", j))
        for s in symbols(c, fixed):
            union(("Q", j), s)
    groups: dict[object, tuple[list[int], list[int]]] = {}
    for i in range(len(kb)):
        groups.setdefault(find(("S", i)), ([], []))[0].append(i)
    for j in range(len(claims)):
        groups.setdefault(find(("Q", j)), ([], []))[1].append(j)
    return list(groups.values())


def _component_task(
    kb: list[Formula], claims: list[Formula], timeout_ms: int, memory_mb: int = 0
) -> tuple[str, list[Verdict | None], str]:
    """Check the consistency of one part of the rulebook, then the claims about it.

    Returns ``(status, verdicts, reason)``. ``"sat"``: the part is consistent and the verdicts are
    computed; ``"unsat"``: the part contradicts itself; ``"unknown"``: Z3 did not decide consistency
    (time, memory or internal error, given in ``reason``) and the claims stay undecided.
    """
    z3.set_param("memory_max_size", memory_mb)  # Z3 memory limit in this process (0: none)
    solver, claims_z3 = _solver(kb, claims, timeout_ms)
    try:
        status = solver.check()
    except z3.Z3Exception as e:
        return "unknown", [None] * len(claims), f"internal error: {e}"
    if status == z3.unsat:
        return "unsat", [], ""
    if status != z3.sat:
        return "unknown", [None] * len(claims), solver.reason_unknown()
    verdicts: list[Verdict | None] = []
    for k in range(len(claims)):
        try:
            verdicts.append(_verdict(solver, claims_z3[k]))
        except z3.Z3Exception:
            # Internal Z3 error (for example "unreachable" after running out of memory): the claim stays
            # undecided and is rebuilt; the solver state is no longer reliable, so a new one is made.
            verdicts.append(None)
            solver, claims_z3 = _solver(kb, claims, timeout_ms)
    return "sat", verdicts, ""


def run_isolated(jobs: list, workers: int, on_crash):
    """Run every job ``(key, function, arguments)`` in its own process, at most ``workers`` at a time.

    Z3 sometimes kills its process outright (for example on an unhandled out-of-memory). In a shared
    process pool this would stop every job; here only the job hit fails, and its result is
    ``on_crash(key)``. A new process per job also frees the memory Z3 leaves behind.
    Yields ``(key, result)`` as jobs finish.
    """
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    # forkserver: every process starts from a clean process (without the threads of the main program),
    # with Z3 and the package already loaded.
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload(["bythebooktest.logic"])
    queue = list(reversed(jobs))
    running: dict = {}  # future -> (key, executor)
    try:
        while queue or running:
            while queue and len(running) < workers:
                key, fn, args = queue.pop()
                ex = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                running[ex.submit(fn, *args)] = (key, ex)
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                key, ex = running.pop(fut)
                try:
                    result = fut.result()
                except BrokenProcessPool:
                    result = on_crash(key)
                ex.shutdown(wait=False)
                yield key, result
    finally:
        for _, ex in running.values():
            ex.shutdown(wait=False, cancel_futures=True)


def _crashed_task(task) -> tuple[str, list, str]:
    return "unknown", [None] * len(task[1]), "process killed by Z3 (out of memory?)"


@dataclass
class SlicedEvaluation:
    """Result of :func:`evaluate_sliced`."""

    consistent: bool
    verdicts: list[Verdict | None]
    inconsistent: list[list[int]]  # parts that contradict themselves (statement indices)
    # Why Z3 did not decide consistency for a group of claims (those claims stay undecided).
    undecided_reasons: list[str] = field(default_factory=list)
    retried: int = 0  # undecided claims checked a second time with more memory
    recovered: int = 0  # of those, the ones decided at the second attempt


def evaluate_sliced(
    kb: list[Formula],
    claims: list[Formula],
    timeout_ms: int = 30000,
    workers: int = 1,
    progress=None,
    chunk: int = 8,
    memory_mb: int = 0,
) -> SlicedEvaluation:
    """Like :func:`evaluate`, but every claim is checked only against the part of the rulebook it shares
    symbols with.

    Statements with no symbol in common with the claim, not even through other statements, cannot
    change its answer. The consistency of every part is checked too.
    """
    tasks: list[tuple[list[int], list[int]]] = []
    for stmts, cls in components(kb, claims):
        if not cls:
            tasks.append((stmts, []))
        for i in range(0, len(cls), chunk):
            tasks.append((stmts, cls[i : i + chunk]))
    verdicts: list[Verdict | None] = [None] * len(claims)
    bad: list[list[int]] = []
    proven: set[tuple[int, ...]] = set()  # parts whose consistency at least one check proved
    unproven: dict[tuple[int, ...], list[int]] = {}
    reasons: list[str] = []
    done = 0
    total = len(claims)

    def collect(task, result) -> None:
        nonlocal done
        stmts, cls = task
        status, vs, reason = result
        if status == "unsat":
            if stmts not in bad:
                bad.append(stmts)
            return
        if status == "sat":
            proven.add(tuple(stmts))
            for j, v in zip(cls, vs):
                verdicts[j] = v
        else:
            # An undecided check does not stop everything: its claims stay undecided and are rebuilt.
            # The consistency of the part is proved by another check, or by the final one.
            unproven[tuple(stmts)] = stmts
            reasons.append(reason)
        done += len(cls)
        if progress is not None and cls:
            progress(done, total)

    if workers <= 1:
        for task in tasks:
            collect(task, _component_task([kb[i] for i in task[0]], [claims[j] for j in task[1]], timeout_ms))
    else:
        jobs = [
            (t, _component_task, ([kb[i] for i in t[0]], [claims[j] for j in t[1]], timeout_ms, memory_mb))
            for t in tasks
        ]
        for task, result in run_isolated(jobs, workers, _crashed_task):
            collect(task, result)

    # Undecided claims in parts without contradictions (time or memory): an undecided claim is not
    # wrong, and rebuilding it would cost a whole check round. Second attempt: one claim per process,
    # with a memory limit four times higher and a quarter of the processes at a time.
    retried = recovered = 0
    if memory_mb and not bad:
        part_of = {j: stmts for stmts, cls in tasks for j in cls}
        retry = [j for j, v in enumerate(verdicts) if v is None]
        retried = len(retry)
        jobs = [
            ((part_of[j], [j]), _component_task, ([kb[i] for i in part_of[j]], [claims[j]], timeout_ms, 4 * memory_mb))
            for j in retry
        ]
        for (stmts, cls), (status, vs, _reason) in run_isolated(jobs, max(1, workers // 4), _crashed_task):
            if status == "unsat":
                if stmts not in bad:
                    bad.append(stmts)
            elif status == "sat":
                proven.add(tuple(stmts))
                verdicts[cls[0]] = vs[0]
                recovered += vs[0] is not None

    # Parts whose consistency no check decided: a last check, alone, with no memory limit and twice the
    # time. If even this does not decide, the rulebook cannot be certified.
    for key, stmts in unproven.items():
        if key in proven or stmts in bad:
            continue
        args: tuple[list[Formula], list[Formula], int, int] = ([kb[i] for i in stmts], [], 2 * timeout_ms, 0)
        if workers > 1:
            job: list[tuple[Any, Callable, tuple]] = [((stmts, []), _component_task, args)]
            _, (status, _, reason) = next(run_isolated(job, 1, _crashed_task))
        else:
            status, _, reason = _component_task(*args)
        if status == "unsat":
            bad.append(stmts)
        elif status != "sat":
            raise RuntimeError(
                f"Z3 did not decide the consistency of a part of the rulebook ({len(stmts)} statements): {reason}"
            )
    return SlicedEvaluation(
        consistent=not bad,
        verdicts=verdicts,
        inconsistent=bad,
        undecided_reasons=reasons,
        retried=retried,
        recovered=recovered,
    )


def inconsistency_core(kb: list[Formula], timeout_ms: int = 60000) -> list[int]:
    """Indices of a set of statements that together contradict each other (an unsatisfiable core)."""
    enc = _Encoder()
    encoded = [enc.encode(f) for f in kb]
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.set("core.minimize", True)
    labels = []
    for i, f in enumerate(encoded):
        label = z3.Bool(f"s{i}")
        labels.append(label)
        solver.assert_and_track(f, label)
    solver.add(enc.distinct())
    if solver.check() != z3.unsat:
        return []
    core = {str(c) for c in solver.unsat_core()}
    return [i for i, label in enumerate(labels) if str(label) in core]


def evaluate(
    kb: list[Formula], claims: list[Formula], timeout_ms: int = 30000, workers: int = 1, progress=None
) -> Evaluation:
    """Compute the expected answer of every claim in classical logic.

    TRUE if the rulebook entails the claim, FALSE if it entails its negation, UNKNOWN otherwise. A
    ``None`` verdict means that Z3 did not decide within the time limit.

    With ``workers > 1`` the claims are split among processes, each with the whole rulebook (the checks
    are independent). ``progress(done, total)`` is called as groups of claims finish.
    """
    solver, claims_z3 = _solver(kb, claims, timeout_ms)
    status = solver.check()
    if status == z3.unsat:
        return Evaluation(consistent=False, verdicts=[None] * len(claims))
    if status != z3.sat:
        raise RuntimeError("Z3 did not decide the consistency of the rulebook within the time limit")

    total = len(claims)
    if workers <= 1 or total < 2 * workers:
        verdicts = []
        for n, c in enumerate(claims_z3, 1):
            verdicts.append(_verdict(solver, c))
            if progress is not None and (n % 10 == 0 or n == total):
                progress(n, total)
        return Evaluation(consistent=True, verdicts=verdicts)

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    chunk = max(1, total // (workers * 4))
    batches = [list(range(i, min(i + chunk, total))) for i in range(0, total, chunk)]
    result: list[Verdict | None] = [None] * total
    done = 0
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(kb, claims, timeout_ms)
    ) as pool:
        futures = [pool.submit(_worker_run, b) for b in batches]
        for fut in as_completed(futures):
            for i, v in fut.result():
                result[i] = v
                done += 1
            if progress is not None:
                progress(done, total)
    return Evaluation(consistent=True, verdicts=result)
