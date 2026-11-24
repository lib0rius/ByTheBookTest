"""The prompts sent to the model.

Two request formats are supported:

- **chat** models get a system prompt with the task and the rules of interpretation, and a user prompt
  with the rulebook followed by the claim. The rulebook comes first and is the same for all claims of a
  rulebook, so the server can read it once and reuse it from its prompt cache;
- **decision** models (Liquid AI "System One" models such as d1-2B) are served by llama.cpp on
  ``/v1/systemone``: the request carries the context (``state``, here the rulebook) and a ``choice``
  question with the options and their description. The server builds the prompt with the model's own
  template and returns the chosen option with the probability of each option.

The knowledge probe asks the same decision models about the real world, with no rulebook.

The texts below are part of the published results: changing any of them changes ``request_sha256``, and
a run with a different fingerprint is a different experiment.
"""

from __future__ import annotations

import hashlib
import json

from bythebooktest.records import KBRecord

SYSTEM_PROMPT = """\
You are a reasoning engine that works on a Knowledge Base.

Decide whether the CLAIM is TRUE, FALSE or UNKNOWN, using only the statements
in the KNOWLEDGE BASE.

The KNOWLEDGE BASE defines the world of this task. It may contradict the real
world, mathematics or common sense. This is intentional. Do not correct it.
Do not use any knowledge that is not written in the KNOWLEDGE BASE.

Answer:
TRUE if the CLAIM follows logically from the KNOWLEDGE BASE.
FALSE if the negation of the CLAIM follows logically from the KNOWLEDGE BASE.
UNKNOWN if neither follows.
The KNOWLEDGE BASE never contradicts itself, so the CLAIM and its negation
never both follow.

Rules of interpretation:
1. Use classical logic. Every valid inference counts, including reasoning
   backwards: if every Tarv is a Pelm and Kiro is not a Pelm, then Kiro is
   not a Tarv.
2. Do not assume anything that the KNOWLEDGE BASE does not state, even if it
   seems obvious. For example, "Kiro lives in Dask" does not imply "Kiro does
   not live in Morv".
3. Different names always refer to different entities.
4. A group is complete only if the KNOWLEDGE BASE says so, for example "The
   only moons of Zeta are Io and Rhea." Otherwise the group may have members
   that are not mentioned.
5. "A or B" means at least one of them, possibly both. "Either A or B, but not
   both" means exactly one of them.
6. "A if and only if B" works in both directions.
7. "Generally", "usually" and "most" say nothing about any specific entity.
8. "Can", "must", "is allowed to" and "does" are different properties. One
   does not imply another unless the KNOWLEDGE BASE says so.
9. "Every X is Y" is TRUE when there is no X at all.
10. Years are numbers. "Before" and "after" a year exclude that year.
    "From one year to another" includes both years.

Your final answer must be exactly one JSON object and nothing else:
{"answer": "TRUE"}, {"answer": "FALSE"} or {"answer": "UNKNOWN"}."""


USER_TEMPLATE = "KNOWLEDGE BASE\n{kb}\n\nCLAIM\n{claim}"


def kb_block(kb: KBRecord) -> str:
    """The rulebook as text: one statement per line, in order."""
    return "\n".join(s.text for s in kb.statements)


def user_prompt(kb: KBRecord, claim_text: str) -> str:
    """The user message for chat models: rulebook first, then the claim."""
    return USER_TEMPLATE.format(kb=kb_block(kb), claim=claim_text)


def messages(kb: KBRecord, claim_text: str) -> list[dict[str, str]]:
    """The chat messages for one claim."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(kb, claim_text)},
    ]


# --- Decision models ("System One") ------------------------------------------------------------

SYSTEMONE_CRITERIA = {  # option -> description, with the same definitions as the system prompt
    "TRUE": "TRUE: the CLAIM follows logically from the KNOWLEDGE BASE.",
    "FALSE": "FALSE: the negation of the CLAIM follows logically from the KNOWLEDGE BASE.",
    "UNKNOWN": "UNKNOWN: neither the CLAIM nor its negation follows from the KNOWLEDGE BASE.",
}


def systemone_instructions(claim_text: str) -> str:
    """The system prompt without its last paragraph on the JSON format: here the answer is an option."""
    rules = SYSTEM_PROMPT.rsplit("\n\nYour final answer", 1)[0]
    return f"{rules}\n\nCLAIM\n{claim_text}"


def systemone_request(kb: KBRecord, claim_text: str) -> dict:
    """The request body for ``/v1/systemone``: the rulebook as state, the claim as a choice question."""
    return {
        "state": f"KNOWLEDGE BASE\n{kb_block(kb)}",
        "questions": {
            "claim": {
                "type": "choice",
                "instructions": systemone_instructions(claim_text),
                "criteria": dict(SYSTEMONE_CRITERIA),
            }
        },
    }


# --- Knowledge probe: the model judges a sentence about the real world, with no rulebook --------

WORLD_STATE = "REAL WORLD\nNo document is given. Use what you know about the real world."
WORLD_INSTRUCTIONS = """\
Decide whether the CLAIM is TRUE or FALSE in the real world, using what you know.
Answer UNKNOWN if you do not know.

CLAIM
{claim}"""
WORLD_CRITERIA = {
    "TRUE": "TRUE: the CLAIM is true in the real world.",
    "FALSE": "FALSE: the CLAIM is false in the real world.",
    "UNKNOWN": "UNKNOWN: you do not know whether the CLAIM is true or false in the real world.",
}


def world_request(claim_text: str) -> dict:
    """The request body for the knowledge probe: no rulebook, the claim judged against the real world."""
    return {
        "state": WORLD_STATE,
        "questions": {
            "claim": {
                "type": "choice",
                "instructions": WORLD_INSTRUCTIONS.format(claim=claim_text),
                "criteria": dict(WORLD_CRITERIA),
            }
        },
    }


def prompt_sha256() -> str:
    """Fingerprint of the chat prompts, recorded with every run."""
    return hashlib.sha256((SYSTEM_PROMPT + "\x00" + USER_TEMPLATE).encode()).hexdigest()


def request_sha256() -> str:
    """Fingerprint of everything sent to the model besides rulebook and claim.

    Covers the chat prompts and the decision-model requests with the text of their options, which
    :func:`prompt_sha256` does not include. Recorded with every run, and checked when a run is resumed.
    """
    empty = KBRecord(kb_id="", level=1, statements=[])
    parts = {
        "system": SYSTEM_PROMPT,
        "user": USER_TEMPLATE,
        "systemone": systemone_request(empty, "{claim}"),
        "world": world_request("{claim}"),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
