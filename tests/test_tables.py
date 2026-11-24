"""Curated tables: format, negations and sizes."""

from pathlib import Path

import pytest

from bythebooktest.tables import load_tables, negate

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"history_geography", "nature", "math_geometry", "common_sense", "implicit_assumptions"}


@pytest.fixture(scope="module")
def tables():
    return load_tables(ROOT / "data" / "tables")


@pytest.mark.parametrize(
    "vp, expected",
    [
        ("is a star", "is not a star"),
        ("was born in Corsica", "was not born in Corsica"),
        ("has eight legs", "does not have eight legs"),
        ("orbits the Earth", "does not orbit the Earth"),
        ("flies", "does not fly"),
        ("watches the sea", "does not watch the sea"),
        ("goes north", "does not go north"),
        ("dies young", "does not die young"),
        ("freezes at 0 degrees", "does not freeze at 0 degrees"),
        ("rises in the east", "does not rise in the east"),
        ("can fly", "cannot fly"),
    ],
)
def test_negate(vp, expected):
    assert negate(vp) == expected


@pytest.mark.parametrize("vp", ["has no rings", "never meets", "painted the Mona Lisa"])
def test_negate_refuses_irregular_or_negative(vp):
    with pytest.raises(ValueError):
        negate(vp)


def test_categories_and_sizes(tables):
    # Level 3 needs about 50 rows per category per rulebook, each row used at most once per rulebook.
    counts = {c: 0 for c in CATEGORIES}
    for r in tables.facts:
        counts[r.category] += 1
    assert all(n >= 50 for n in counts.values()), counts


def test_implicit_assumption_rows_have_traps(tables):
    for r in tables.facts:
        assert bool(r.traps) == (r.category == "implicit_assumptions"), r.id


def test_abilities_cover_both_world_values_and_two_categories(tables):
    assert {a.world_can for a in tables.abilities} == {True, False}
    assert {a.world_does for a in tables.abilities} == {True, False}
    assert {a.category for a in tables.abilities} == {"nature", "common_sense"}


def test_no_subject_predicate_pair_in_two_rows(tables):
    # The same fact in two rows would let one claim prove it and another deny it.
    seen: dict[tuple[str, str], str] = {}
    for r in tables.facts:
        for vp in [r.true, *r.false, *r.traps]:
            key = (r.subject.lower(), vp.pos.lower())
            assert key not in seen, (r.id, seen.get(key), key)
            seen[key] = r.id


def test_no_negative_words_in_derived_positives(tables):
    for r in tables.facts:
        for vp in [r.true, *r.false, *r.traps]:
            assert vp.pos != vp.neg, r.id
            assert " not not " not in f" {vp.neg} ", r.id
