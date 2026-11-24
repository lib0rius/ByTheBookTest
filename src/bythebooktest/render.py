"""Sentence templates: from the structured form of a statement to an English sentence.

Every sentence of a rulebook and every claim comes from one of these functions, so reviewing the
language of the whole dataset means reviewing this file.
"""

from __future__ import annotations

import inflect

_ENGINE = inflect.engine()


def with_article(noun: str) -> str:
    """Add the indefinite article: "penguin" -> "a penguin", "eagle" -> "an eagle"."""
    return _ENGINE.a(noun)


def plural_invented(noun: str) -> str:
    """Regular plural for invented words (inflect would apply irregular rules, for example to -man)."""
    return noun + ("es" if noun.endswith(("s", "x", "z", "sh", "ch")) else "s")


def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def sentence(text: str) -> str:
    text = capitalize_first(text.strip())
    return text if text.endswith(".") else text + "."


def fact(subject: str, vp: str) -> str:
    """A fact: "Napoleon" + "was born in Australia" -> "Napoleon was born in Australia."."""
    return sentence(f"{subject} {vp}")


def rule_class(noun: str, vp: str) -> str:
    """A rule about a class, as in "Every kelvar was born in Australia"."""
    return sentence(f"every {noun} {vp}")


def rule_class_negative(noun: str, pos_vp: str) -> str:
    """A rule with a negated conclusion, as in "No kelvar was born in Australia"."""
    return sentence(f"no {noun} {pos_vp}")


def rule_generic(premise_vp: str, conclusion_vp: str, person: bool) -> str:
    """A rule with any premise, as in "Everyone who was born in Japan is a kelvar"."""
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {premise_vp} {conclusion_vp}")


def rule_generic_negative(premise_vp: str, pos_vp: str, person: bool) -> str:
    """A rule with any premise and a negated conclusion: "No one who was born in Japan is a kelvar."."""
    head = "no one who" if person else "nothing that"
    return sentence(f"{head} {premise_vp} {pos_vp}")


def rule_two_premises(
    first_noun: str | None, premise_vps: list[str], connective: str, conclusion_vp: str, person: bool
) -> str:
    """A rule with several conditions.

    "Every kelvar that is not a trusap was born in Australia."
    "Everything that is a kelvar or is a trusap was born in Australia."
    """
    if first_noun is not None and connective == "and" and len(premise_vps) == 2:
        return sentence(f"every {first_noun} that {premise_vps[1]} {conclusion_vp}")
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {f' {connective} '.join(premise_vps)} {conclusion_vp}")


def compound(vp1: str, vp2: str, op: str) -> str:
    """Two phrases joined by "or" (at least one) or "either ... or ..., but not both" (rule 5 of the prompt)."""
    if op == "or":
        return f"{vp1} or {vp2}"
    return f"either {vp1} or {vp2}, but not both"


def fact_compound(subject: str, vp1: str, vp2: str, op: str) -> str:
    """A compound fact: "Napoleon either is a kelvar or was born in Australia, but not both."."""
    return sentence(f"{subject} {compound(vp1, vp2, op)}")


def rule_compound(noun: str | None, premise_vp: str, vp1: str, vp2: str, op: str, person: bool) -> str:
    """A rule with a compound conclusion: "Every kelvar was born in Australia or is a trusap."."""
    if noun is not None:
        return sentence(f"every {noun} {compound(vp1, vp2, op)}")
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {premise_vp} {compound(vp1, vp2, op)}")


def iff(vp_a: str, vp_b: str, person: bool) -> str:
    """A biconditional: "Someone is a kelvar if and only if that person was born in Australia." (rule 6)."""
    if person:
        return sentence(f"someone {vp_a} if and only if that person {vp_b}")
    return sentence(f"something {vp_a} if and only if it {vp_b}")


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def join_names(names: list[str]) -> str:
    """["A", "B", "C"] -> "A, B and C"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def rel_fact(subject: str, phrase: str, obj: str) -> str:
    """A relation between two entities, as in "Napoleon is linked to Kelvorax"."""
    return sentence(f"{subject} {phrase} {obj}")


def rel_rule(phrase: str, noun: str, conclusion_vp: str, person: bool, negative_pos_vp: str | None = None) -> str:
    """A rule over a relation: "Everyone who is linked to a trusap was born in Australia."."""
    obj = with_article(noun)
    if negative_pos_vp is not None:
        head = "no one who" if person else "nothing that"
        return sentence(f"{head} {phrase} {obj} {negative_pos_vp}")
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {phrase} {obj} {conclusion_vp}")


def rel3_rule(
    phrase1: str, phrase2: str, noun: str, conclusion_vp: str, person: bool, negative_pos_vp: str | None = None
) -> str:
    """A rule over three entities: "Everything that is linked to something that is close to a trusap
    was born in Australia."
    """
    middle = f"{phrase1} something that {phrase2} {with_article(noun)}"
    if negative_pos_vp is not None:
        head = "no one who" if person else "nothing that"
        return sentence(f"{head} {middle} {negative_pos_vp}")
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {middle} {conclusion_vp}")


def only_rule(premise_vp: str, phrase: str, plural: str, person: bool) -> str:
    """An "only" rule: "Everyone who was born in Australia is linked to only trusaps."."""
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {premise_vp} {phrase} only {plural}")


def lives(subject: str, place: str) -> str:
    return sentence(f"{subject} lives in {place}")


def scoped_rule(singular: str, place: str, vp: str, negative: bool = False) -> str:
    """A rule about a set: "Every penguin that lives in Zorbak can fly." or "No penguin that ..."."""
    head = "no" if negative else "every"
    return sentence(f"{head} {singular} that lives in {place} {vp}")


def closed_list(singular: str, plural: str, place: str, members: list[str]) -> str:
    """A closed list, as in "The only penguins that live in Zorbak are A, B and C"."""
    if not members:
        return sentence(f"no {singular} lives in {place}")
    if len(members) == 1:
        return sentence(f"the only {singular} that lives in {place} is {members[0]}")
    return sentence(f"the only {plural} that live in {place} are {join_names(members)}")


def set_claim(kind: str, n: int, singular: str, plural: str, place: str, vp: str) -> str:
    """A claim about a set; ``vp`` must work in singular and plural ("can fly").

    "Some penguin that lives in Zorbak can fly." / "Every penguin that lives in Zorbak can fly."
    "Exactly two penguins that live in Zorbak can fly." / "At least one penguin that lives in ..."
    """
    if kind == "exists":
        return sentence(f"some {singular} that lives in {place} {vp}")
    if kind == "universal":
        return sentence(f"every {singular} that lives in {place} {vp}")
    head = "exactly" if kind == "exactly" else "at least"
    if n == 1:
        return sentence(f"{head} one {singular} that lives in {place} {vp}")
    return sentence(f"{head} {NUMBER_WORDS[n]} {plural} that live in {place} {vp}")


def year_fact(subject: str, verb: str, year: int) -> str:
    """A dated fact: "Napoleon was born in 1975."."""
    return sentence(f"{subject} {verb} in {year}")


def year_rule(
    verb: str,
    op: str,
    a: int,
    b: int | None,
    conclusion_vp: str,
    person: bool,
    negative_pos_vp: str | None = None,
    scope_noun: str | None = None,
) -> str:
    """A rule about dates: "Every kelvar that was born after 1970 is young." (rule 10 of the prompt).

    With ``scope_noun`` the rule holds for one class only; without it, for anyone with a date.
    """
    when = {"after": f"after {a}", "before": f"before {a}", "interval": f"from {a} to {b}"}[op]
    if scope_noun is not None:
        if negative_pos_vp is not None:
            return sentence(f"no {scope_noun} that {verb} {when} {negative_pos_vp}")
        return sentence(f"every {scope_noun} that {verb} {when} {conclusion_vp}")
    if negative_pos_vp is not None:
        head = "no one who" if person else "nothing that"
        return sentence(f"{head} {verb} {when} {negative_pos_vp}")
    head = "everyone who" if person else "everything that"
    return sentence(f"{head} {verb} {when} {conclusion_vp}")


def generally(plural: str, verb: str) -> str:
    """A "generally" statement: "Penguins generally fly." It says nothing about any single penguin (rule 7)."""
    return sentence(f"{plural} generally {verb}")
