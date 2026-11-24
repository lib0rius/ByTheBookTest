"""The reasoning groups available to the generator, by name."""

from __future__ import annotations

from .groups import DirectFact, Group, Modal, RuleChain
from .groups_l2 import L2_GROUPS
from .groups_l3 import L3_GROUPS

GROUPS: dict[str, Group] = {g.name: g for g in (DirectFact(), RuleChain(), Modal(), *L2_GROUPS, *L3_GROUPS)}
