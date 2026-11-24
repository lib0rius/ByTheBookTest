"""Sentence templates."""

from bythebooktest import render
from bythebooktest.kb import can, does, membership, must


def test_fact_capitalizes_subject():
    assert render.fact("the Sun", "is a planet") == "The Sun is a planet."


def test_articles():
    assert membership("eagle").pos == "is an eagle"
    assert membership("penguin").neg == "is not a penguin"


def test_rules():
    assert render.rule_class("penguin", can("fly").pos) == "Every penguin can fly."
    assert render.rule_generic("was born in Japan", "is a kelvar", person=True) == (
        "Everyone who was born in Japan is a kelvar."
    )
    assert render.rule_generic("is hot", "is a kelvar", person=False) == ("Everything that is hot is a kelvar.")


def test_modal_negations():
    assert can("fly").neg == "cannot fly"
    assert does("fly", "flies").pos == "flies"
    assert does("fly", "flies").neg == "does not fly"
    assert must("sing").neg == "does not have to sing"


def test_generally():
    assert render.generally("Penguins", "fly") == "Penguins generally fly."


def test_level3_templates():
    assert render.rel_rule("is linked to", "trusap", "was born in Australia", person=True) == (
        "Everyone who is linked to a trusap was born in Australia."
    )
    assert render.rel3_rule("is linked to", "is close to", "trusap", "is hot", person=False) == (
        "Everything that is linked to something that is close to a trusap is hot."
    )
    assert render.only_rule("was born in Australia", "is linked to", "kelvars", person=True) == (
        "Everyone who was born in Australia is linked to only kelvars."
    )
    assert render.closed_list("penguin", "penguins", "Zorbak", ["Ada", "Bo", "Cy"]) == (
        "The only penguins that live in Zorbak are Ada, Bo and Cy."
    )
    assert render.closed_list("penguin", "penguins", "Zorbak", []) == "No penguin lives in Zorbak."
    assert render.set_claim("exactly", 2, "penguin", "penguins", "Zorbak", "can fly") == (
        "Exactly two penguins that live in Zorbak can fly."
    )
    assert render.set_claim("exists", 0, "penguin", "penguins", "Zorbak", "can fly") == (
        "Some penguin that lives in Zorbak can fly."
    )
    assert render.scoped_rule("penguin", "Zorbak", "can fly", negative=True) == (
        "No penguin that lives in Zorbak can fly."
    )


def test_level2_templates():
    assert render.rule_class_negative("kelvar", "was born in Australia") == ("No kelvar was born in Australia.")
    assert render.rule_two_premises(
        "kelvar", ["is a kelvar", "is not a trusap"], "and", "was born in Australia", person=True
    ) == ("Every kelvar that is not a trusap was born in Australia.")
    assert render.rule_two_premises(None, ["is a kelvar", "is a trusap"], "or", "is hot", person=False) == (
        "Everything that is a kelvar or is a trusap is hot."
    )
    assert render.fact_compound("Napoleon", "is a kelvar", "was born in Australia", "xor") == (
        "Napoleon either is a kelvar or was born in Australia, but not both."
    )
    assert (
        render.rule_compound("kelvar", "is a kelvar", "is hot", "is a trusap", "or", person=False)
        == "Every kelvar is hot or is a trusap."
    )
    assert render.iff("is a kelvar", "was born in Australia", person=True) == (
        "Someone is a kelvar if and only if that person was born in Australia."
    )
    assert render.year_rule("was born", "after", 1970, None, "is a kelvar", person=True) == (
        "Everyone who was born after 1970 is a kelvar."
    )
    assert render.year_rule("was registered", "interval", 1990, 2000, "is old", person=False) == (
        "Everything that was registered from 1990 to 2000 is old."
    )
