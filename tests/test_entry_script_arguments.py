# Copyright 2026 NVIDIA. Apache-2.0.
"""The launcher, the argument surface and the reward's config must be one thing.

Three files have to agree before a single optimizer step can run, and they are in three
different languages of description:

    training/grpo/config.py                 builds the command line from an arm's YAML
    trl_patch/scripts/utils.py              declares what that command line may contain
    trl_patch/rewards/self_saliency.py      declares what its `configure()` accepts

Nothing made them agree. `docs/reproduce.md` and the six arm configs are checked against
the published checkpoints by `test_configs_match_paper_checkpoints.py`, but that stops at
"config.py emits the right VALUES". Whether anything downstream would ACCEPT them was
untested, and all three seams were broken at once after the port:

  * `--overlap_metric phi` -- what config.py emits for five of the six arms -- was not
    among the declared choices, which were still the research tree's `mean_in`,
    `mean_in_v2`, `auroc`, `logratio`. argparse rejects an invalid choice, so five arms
    died at argument parsing.
  * the entry script passed `configure()` eight keywords it does not have (`null_offsets`,
    `logratio_clip`, `rect_seed`, `chain_boxes`, ...), and `configure()` raises KeyError
    on an unknown setting rather than ignoring it.
  * `ScriptArguments` still declared 36 flags for arms that are not in this repository.

Each of those is silent in the one place people look: the config files themselves read
correctly, and `--dry-run` prints a plausible command. The failure is at launch, on eight
GPUs, after the model has loaded.

Everything here is AST-level and needs no torch, no trl and no installed `selfsal`, so it
runs in the bare CPU environment. One test additionally does a real argparse pass when
transformers is importable, which is what catches a TYPE mismatch as opposed to a name or
a choice one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "training/grpo/trl_patch"
UTILS = PATCH / "scripts/utils.py"
ENTRY = PATCH / "grpo_vlm_qwen3.py"
REWARD = PATCH / "rewards/self_saliency.py"

#: Fields `ScriptArguments` inherits from upstream TRL and shares with its other example
#: scripts. They are exempt from "this repository must read it", because it is not this
#: repository's script that reads them.
UPSTREAM_FIELDS = {
    "dataset_name", "dataset_config", "dataset_train_split", "dataset_test_split",
    "dataset_streaming", "gradient_checkpointing_use_reentrant", "ignore_bias_buffers",
}


# ---------------------------------------------------------------------------
# reading the three files
# ---------------------------------------------------------------------------

def _script_argument_fields() -> dict[str, dict]:
    """{field name: {"choices": [...] | None}} for ScriptArguments, from its source."""
    tree = ast.parse(UTILS.read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "ScriptArguments")
    out: dict[str, dict] = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        choices = None
        if isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if kw.arg == "metadata" and isinstance(kw.value, ast.Dict):
                    for k, v in zip(kw.value.keys, kw.value.values):
                        if isinstance(k, ast.Constant) and k.value == "choices":
                            choices = [c.value for c in v.elts]
        out[node.target.id] = {"choices": choices}
    return out


def _entry_script_reads() -> set[str]:
    """Every `script_args.X` the entry script touches."""
    tree = ast.parse(ENTRY.read_text())
    return {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "script_args"
    }


def _configure_keywords() -> tuple[set[str], int]:
    """(keywords passed to the saliency reward's configure(), its line number)."""
    tree = ast.parse(ENTRY.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.startswith("configure")):
            return {kw.arg for kw in node.keywords if kw.arg}, node.lineno
    pytest.fail("the entry script makes no configure() call for the saliency reward")


def _cfg_keys() -> set[str]:
    """Keys of `_CFG` in the saliency reward -- exactly what configure() will accept."""
    tree = ast.parse(REWARD.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_CFG"
                and isinstance(node.value, ast.Dict)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    pytest.fail("could not find the _CFG literal in the saliency reward")


def _flag_values(flags: list[str]) -> dict[str, list[str]]:
    """{--flag: [its values]} for a command line, tolerating nargs and store_true."""
    out: dict[str, list[str]] = {}
    current = None
    for token in flags:
        if token.startswith("--"):
            current = token
            out.setdefault(current, [])
        elif current is not None:
            out[current].append(token)
    return out


#: Imported rather than listed, so an arm added to the paper is covered here by default.
from training.grpo.config import ARMS  # noqa: E402


@pytest.fixture(scope="module")
def cfgmod():
    from training.grpo import config
    return config


@pytest.fixture(scope="module")
def fields():
    return _script_argument_fields()


# ---------------------------------------------------------------------------
# the launcher against the argument surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_emitted_values_are_declared_choices(arm, cfgmod, fields):
    """Every value config.py emits for a constrained flag must be one of its choices.

    This is the one that was failing. argparse enforces `choices`, so a flag whose
    declared set has moved on from what the launcher emits is not a warning -- it is
    SystemExit(2) before the trainer is constructed.
    """
    flags = _flag_values(cfgmod.training_flags(cfgmod.load_arm(arm), "/tmp/out"))
    problems = []
    for flag, values in flags.items():
        name = flag.lstrip("-")
        spec = fields.get(name)
        if spec is None or spec["choices"] is None:
            continue
        for value in values:
            if value not in spec["choices"]:
                problems.append(
                    f"{flag} {value!r} is not in {spec['choices']}")
    assert not problems, f"{arm}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("arm", ARMS)
def test_emitted_command_line_parses(arm, cfgmod):
    """A real argparse pass over each arm, which also catches a TYPE mismatch.

    Only `ScriptArguments` is parsed; the GRPOConfig and ModelConfig flags land in the
    remainder, because bringing those in would need an installed TRL and this has to run
    on a bare clone. `test_emitted_values_are_declared_choices` is the version of this
    that never skips.
    """
    import contextlib
    import importlib.util
    import io

    pytest.importorskip("transformers")
    from transformers import HfArgumentParser

    spec = importlib.util.spec_from_file_location("_patch_scripts_utils", UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    flags = cfgmod.training_flags(cfgmod.load_arm(arm), "/tmp/out")
    parser = HfArgumentParser((module.ScriptArguments,))
    try:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            parser.parse_args_into_dataclasses(flags, return_remaining_strings=True)
    except SystemExit:
        pytest.fail(f"{arm}: the emitted command line does not parse:\n"
                    f"{err.getvalue().strip().splitlines()[-1] if err.getvalue() else ''}")


# ---------------------------------------------------------------------------
# the argument surface against the entry script
# ---------------------------------------------------------------------------

def test_every_attribute_the_entry_script_reads_is_declared(fields):
    """`script_args.X` must be a field, or it is an AttributeError at launch."""
    undeclared = sorted(_entry_script_reads() - set(fields))
    assert not undeclared, (
        f"the entry script reads {undeclared} off script_args, but ScriptArguments "
        f"declares no such field")


def test_every_declared_field_is_read(fields):
    """And the reverse: no flag may exist that nothing reads.

    A flag nobody reads is worse than clutter -- it is accepted on the command line and
    silently does nothing, so a run can be named after an arm it did not configure. This
    is how 36 of them accumulated.
    """
    read = _entry_script_reads()
    orphans = sorted(set(fields) - read - UPSTREAM_FIELDS)
    assert not orphans, (
        f"ScriptArguments declares {orphans}, which the entry script never reads. Either "
        f"wire them up or drop them; a flag that is accepted and ignored is a run that "
        f"claims to be something it is not")


# ---------------------------------------------------------------------------
# the entry script against the reward's configuration
# ---------------------------------------------------------------------------

def test_configure_is_passed_only_settings_it_has():
    """configure() RAISES on an unknown setting, so a stale keyword is a hard failure.

    It raises deliberately -- silently accepting `rect_seed=0` would mean the run's own
    configuration call could not be read back as a description of the run. The cost is
    that the entry script's call and `_CFG` have to be kept in step, which is this.
    """
    passed, lineno = _configure_keywords()
    accepted = _cfg_keys()
    unknown = sorted(passed - accepted)
    assert not unknown, (
        f"grpo_vlm_qwen3.py:{lineno} passes configure() {unknown}, which are not keys of "
        f"_CFG in rewards/self_saliency.py. configure() raises KeyError on each, on every "
        f"rank, at launch")


def test_every_region_source_the_configs_use_reaches_configure(cfgmod):
    """Each arm's `regions.source` must have a keyword that carries it to the reward.

    The three sources are selected by the ABSENCE of the other two's keywords rather than
    by a name, so a dropped keyword does not fail -- it silently demotes the arm to
    per-step grounding, which is the headline arm. center_rect and question_boxes would
    then both be SELF-SALIENCY under another name, and would still train, finish, and
    disagree with Table 5.
    """
    passed, _ = _configure_keywords()
    carrier = {"dino_per_step": "box_threshold",
               "center_rect": "rect_frac",
               "question_boxes": "question_boxes"}
    used = {(cfgmod.load_arm(a)["reward"]["saliency"].get("regions") or {}).get("source")
            for a in cfgmod.ARMS}
    for source in sorted(s for s in used if s in carrier):
        assert carrier[source] in passed, (
            f"an arm config uses regions.source={source!r}, but the entry script never "
            f"passes configure() its {carrier[source]!r} keyword")
