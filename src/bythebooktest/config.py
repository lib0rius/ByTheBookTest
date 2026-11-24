"""Configuration files, validated with Pydantic.

Two kinds of TOML files drive the commands:

- a **generator** file (``configs/level*.toml``) describes the rulebooks and claims of each level;
- a **run** file (``configs/d1-2b.toml``) describes the model, the inference server and the datasets.

Unknown keys are rejected, so a typo fails loudly instead of being ignored. Relative paths in the files
are relative to the working directory (normally the repository root), where the commands are run.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def read_toml(path: Path) -> dict:
    """Read a TOML file, with a clear message if it is missing."""
    if not path.is_file():
        raise SystemExit(f"{path}: configuration file not found")
    return tomllib.loads(path.read_text(encoding="utf-8"))


# --- Generator ------------------------------------------------------------------------------------


class LevelConfig(_Strict):
    """How to build the rulebooks and claims of one level."""

    kbs: int = Field(gt=0)
    """Number of rulebooks."""
    claims_per_cell: int = Field(gt=0)
    """Claims for each reasoning group, split among the rulebooks."""
    noise_ratio: float = Field(ge=0, lt=1)
    """Share of statements that no claim needs (they make the rulebook longer)."""
    groups: list[str]
    """Reasoning groups used at this level (see :data:`bythebooktest.registry.GROUPS`)."""
    noise_style: Literal["isolated", "similar"] = "isolated"
    """``isolated``: noise about unrelated entities; ``similar``: noise that looks like the proofs."""
    check: Literal["full", "sliced"] = "full"
    """``full``: Z3 checks each claim against the whole rulebook; ``sliced``: only against the
    statements that share a symbol with the claim (equivalent, and much faster on long rulebooks)."""
    z3_timeout_s: int = Field(default=30, gt=0)
    """Time limit for one Z3 check."""
    chunk: int = Field(default=8, gt=0)
    """Claims per worker process in the sliced check."""
    z3_memory_mb: int = Field(default=0, ge=0)
    """Memory limit for Z3 in each worker process; 0 means no limit."""

    @field_validator("groups")
    @classmethod
    def _known_groups(cls, groups: list[str]) -> list[str]:
        from .registry import GROUPS

        unknown = [g for g in groups if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown reasoning groups: {unknown}; available: {sorted(GROUPS)}")
        return groups


class GeneratorConfig(_Strict):
    """A generator file: a seed and one table per level, such as ``[level.2]``."""

    seed: int
    level: dict[int, LevelConfig]
    _raw: dict = PrivateAttr(default_factory=dict)

    @property
    def levels(self) -> dict[int, LevelConfig]:
        return self.level

    @property
    def raw(self) -> dict:
        """The file as written, recorded in the dataset manifest and in the checkpoint key."""
        return self._raw

    @classmethod
    def load(cls, path: Path) -> GeneratorConfig:
        return cls.from_dict(read_toml(path))

    @classmethod
    def from_dict(cls, raw: dict) -> GeneratorConfig:
        """Validate a configuration already read, for example from a dataset manifest."""
        config = cls.model_validate(raw)
        config._raw = raw
        return config


# --- Test run ---------------------------------------------------------------------------------------


class ModelSpec(_Strict):
    """A model to test."""

    name: str
    """Name used in result files, for example ``"d1-2B"``."""
    path: str
    """The GGUF file."""
    kind: Literal["chat", "systemone"] = "chat"
    """``chat``: a chat model answering in JSON, possibly after thinking; ``systemone``: a decision model
    served on ``/v1/systemone``, which returns one of the options with their probabilities."""


class ServerOptions(_Strict):
    """How to start ``llama-server`` from llama.cpp."""

    binary: str = "llama-server"
    """The executable, by path or by name on ``PATH``."""
    ctx_size: int = Field(default=131072, gt=0)
    """Context per request, in tokens; the server gets ``ctx_size * parallel`` in total."""
    parallel: int = Field(default=1, gt=0)
    """Requests served at the same time (server slots)."""
    reasoning_budget: int = Field(default=8192, ge=0)
    """Maximum thinking tokens for chat models with reasoning."""
    reasoning_budget_message: str = ""
    """Sentence inserted before the end of the thinking when the budget runs out."""
    reasoning: bool = True
    """Whether the model thinks before answering; False for decision models."""
    gpu_layers: int = 999
    require_gpu: bool = True
    """Refuse to start without a visible GPU: llama-server would silently fall back to the CPU."""
    host: str = "127.0.0.1"
    port: int = 8091
    extra_args: list[str] = Field(default_factory=list)
    """Further ``llama-server`` arguments, passed as they are."""

    def command(self, model: Path) -> list[str]:
        """The full ``llama-server`` command line for ``model``."""
        return [
            self.binary,
            "-m",
            str(model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            # -c is the total context, shared among the slots: each slot must get ctx_size in full
            "-c",
            str(self.ctx_size * self.parallel),
            "-np",
            str(self.parallel),
            "-ngl",
            str(self.gpu_layers),
            "--jinja",
            "--metrics",
            "--no-webui",
            *(
                [
                    "--reasoning",
                    "on",
                    "--reasoning-budget",
                    str(self.reasoning_budget),
                    "--reasoning-format",
                    "deepseek",
                ]
                if self.reasoning
                else ["--reasoning", "off"]
            ),
            *(
                ["--reasoning-budget-message", self.reasoning_budget_message]
                if self.reasoning and self.reasoning_budget_message
                else []
            ),
            *self.extra_args,
        ]


class RunConfig(_Strict):
    """A run file: which models answer which claims, and how.

    Everything except ``out`` is part of the run key stored in ``run.json``: a run can only be resumed
    with the same configuration, prompts and datasets.
    """

    run_id: str
    """Name of the run folder under ``out``."""
    out: str
    """Folder where run folders are created."""
    datasets: list[str]
    """Dataset folders, each with ``kbs.jsonl`` and ``claims.jsonl``."""
    models: list[ModelSpec] = Field(min_length=1)
    seed: int = 42
    """Sampling seed sent with every chat request."""
    sampling: dict = Field(default_factory=dict)
    """Sampling parameters for chat models (for example ``temperature``), sent as they are."""
    server: ServerOptions = Field(default_factory=ServerOptions)
    answer_tokens: int = Field(default=1024, gt=0)
    """Tokens allowed for the answer after the thinking."""
    claim_timeout_s: float = Field(default=1800.0, gt=0)
    """Time limit for one claim; on timeout the server is restarted and the claim recorded as such."""
    claims_per_kb: int | None = Field(default=None, gt=0)
    """Test only this many claims per rulebook (stratified sample); None for all."""
    kbs: list[str] | None = None
    """Test only these rulebooks; None for all."""
    levels: list[int] | None = None
    """Test only these levels; None for all."""
    fraction: float | None = Field(default=None, gt=0, le=1)
    """Test only this share of the claims (stratified sample); None for all."""
    subset_sampling: Literal["stratified"] = "stratified"
    """How subsets are drawn; recorded in the run key."""
    task: Literal["kb", "world"] = "kb"
    """``kb``: judge the claim against the rulebook (the benchmark); ``world``: judge it against the real
    world, with no rulebook (the knowledge probe, decision models only)."""

    @model_validator(mode="after")
    def _consistent(self) -> RunConfig:
        if self.fraction is not None and self.claims_per_kb is not None:
            raise ValueError("use either fraction or claims_per_kb, not both")
        if self.task == "world" and any(m.kind != "systemone" for m in self.models):
            raise ValueError('task = "world" is only supported for decision models (kind = "systemone")')
        return self

    @classmethod
    def load(
        cls, path: Path, models: list[str] | None = None, parallel: int | None = None, **overrides: object
    ) -> RunConfig:
        """Read a run file and apply command-line overrides (``None`` values are ignored).

        Args:
            path: The run file.
            models: Keep only the models whose name ends with one of these (for example ``["2B"]``).
            parallel: Number of server slots.
            **overrides: Any other field, replaced as given.
        """
        raw = read_toml(path)
        if models:
            raw["models"] = [m for m in raw.get("models", []) if any(m.get("name", "").endswith(k) for k in models)]
            if not raw["models"]:
                raise SystemExit(f"no model matches {models}")
        if parallel is not None:
            raw.setdefault("server", {})["parallel"] = parallel
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(raw)

    def payload_base(self) -> dict:
        """Fields sent with every chat request besides the messages."""
        return {
            **self.sampling,
            "seed": self.seed,
            "max_tokens": self.server.reasoning_budget + self.answer_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }
