# Copyright 2026 NVIDIA. Apache-2.0.
"""The judge's endpoint default and its model-name default move together, never alone.

A gateway addresses the same model by a different name. OpenAI's public API wants the
bare `gpt-4o-mini`; the NVIDIA inference gateway wants the provider-prefixed
`azure/openai/gpt-4o-mini` and answers the bare alias with 403 key_model_access_denied.
So the two defaults are one setting in two variables, and changing either one alone
produces a configuration that authenticates, runs, and fails every judged sample.

The failure is quiet in the direction that matters. A judged training sample that cannot
be scored is MASKED to its group mean (`selfsal/judge.py`), and an evaluation benchmark
that cannot reach its judge falls back to exact matching -- so a mismatched pair does not
crash, it silently removes R_llm from the reward and silently under-reports the
benchmarks. Nobody reads a log line about it three hours into a GPU job.

Six sites carry the pair, in three spellings, which is why this is a test and not a
convention: `selfsal/judge.py` and the EASE reward use the openai SDK's OPENAI_BASE_URL /
JUDGE_MODEL; `baselines/ease/run.sh` re-exports JUDGE_MODEL over the reward's own default;
and the two evaluation launchers use lmms-eval's OPENAI_API_URL / MODEL_VERSION, whose URL
carries the full `/chat/completions` path. `baselines/ease/run.sh` is the reason for the
"every site agrees" rule rather than a per-file rule: it sets the model and NOT the
endpoint, so its default is only correct relative to the endpoint someone else defaults.

docs/install.md section 4 is the reader-facing version of this.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The openai SDK's own two variables, as read by `os.environ.get(NAME, "default")`.
#: ANCHORED AT THE OPENING QUOTE, and the shell ones behind a non-name character: an
#: unanchored `JUDGE_MODEL"` also matches `EASE_JUDGE_MODEL"`, so renaming the variable
#: would leave this whole file passing while reading a setting nothing consumes. That is
#: what mutating the name found; `test_the_site_still_declares_its_defaults` is the test
#: that then goes red instead.
_PY_URL = re.compile(r'"OPENAI_BASE_URL"\s*,\s*"([^"]+)"')
_PY_MODEL = re.compile(r'"JUDGE_MODEL"\s*,\s*"([^"]+)"')
#: The shell spelling, `NAME=${NAME:-default}`.
_SH_JUDGE_MODEL = re.compile(r'(?<![A-Z0-9_])JUDGE_MODEL=\$\{JUDGE_MODEL:-([^}]+)\}')
_SH_EVAL_URL = re.compile(r'(?<![A-Z0-9_])OPENAI_API_URL=\$\{OPENAI_API_URL:-([^}]+)\}')
_SH_EVAL_MODEL = re.compile(r'(?<![A-Z0-9_])MODEL_VERSION=\$\{MODEL_VERSION:-([^}]+)\}')

#: rel path -> (endpoint pattern or None, model pattern). None means the file defaults a
#: model but inherits its endpoint, which is exactly why every site has to agree.
SITES = {
    "selfsal/judge.py": (_PY_URL, _PY_MODEL),
    "baselines/ease/reward_function/judged_perception.py": (_PY_URL, _PY_MODEL),
    "experiments/trained_model/audit.py": (_PY_URL, _PY_MODEL),
    "baselines/ease/run.sh": (None, _SH_JUDGE_MODEL),
    "evaluation/submit.sh": (_SH_EVAL_URL, _SH_EVAL_MODEL),
    "evaluation/run_suite.sh": (_SH_EVAL_URL, _SH_EVAL_MODEL),
}

#: `selfsal/steps/make_data.py` is deliberately absent. It distils the step classifier's
#: labels from Gemini 2.5 Pro, which no OpenAI endpoint serves, so its pair is a matched
#: gateway pair by design rather than drift. docs/publishing.md section 2 records that.

OPENAI_HOST = "api.openai.com"


def _defaults(rel):
    """(endpoint, model) as this file defaults them; either may be None."""
    url_re, model_re = SITES[rel]
    text = (ROOT / rel).read_text()
    url = None
    if url_re is not None:
        m = url_re.search(text)
        assert m, f"{rel}: no endpoint default matched {url_re.pattern!r}"
        url = m.group(1)
    m = model_re.search(text)
    assert m, f"{rel}: no model default matched {model_re.pattern!r}"
    return url, m.group(1)


@pytest.mark.parametrize("rel", sorted(SITES))
def test_the_site_still_declares_its_defaults(rel):
    """A renamed variable would make every check below vacuous rather than red."""
    url, model = _defaults(rel)
    assert model, f"{rel}: empty model default"
    if SITES[rel][0] is not None:
        assert url, f"{rel}: empty endpoint default"


@pytest.mark.parametrize("rel", sorted(SITES))
def test_endpoint_and_model_are_a_matched_pair(rel):
    """Bare model name against OpenAI; provider-prefixed against anything else."""
    url, model = _defaults(rel)
    if url is None:
        pytest.skip(f"{rel} defaults a model and inherits its endpoint; "
                    f"test_every_site_agrees covers it")
    host = urlparse(url).netloc
    prefixed = "/" in model
    if host == OPENAI_HOST:
        assert not prefixed, (
            f"{rel}: endpoint is {host} but the model default {model!r} is "
            f"provider-prefixed. OpenAI's API does not know that name; every judged "
            f"sample would fail. Move the endpoint and the model together.")
    else:
        assert prefixed, (
            f"{rel}: endpoint is {host}, a gateway, but the model default {model!r} is "
            f"the bare alias, which a gateway answers with 403 key_model_access_denied. "
            f"Move the endpoint and the model together.")


def test_every_site_agrees():
    """One setting, six files. A site that disagrees overrides a correct pair into a
    broken one -- `baselines/ease/run.sh` re-exports JUDGE_MODEL over the reward's own
    default, so it can break a file it does not contain."""
    models = {rel: _defaults(rel)[1] for rel in SITES}
    assert len(set(models.values())) == 1, (
        "the judge model default differs between sites:\n  "
        + "\n  ".join(f"{rel}: {m}" for rel, m in sorted(models.items())))

    hosts = {rel: urlparse(_defaults(rel)[0]).netloc
             for rel in SITES if SITES[rel][0] is not None}
    assert len(set(hosts.values())) == 1, (
        "the judge endpoint default differs between sites:\n  "
        + "\n  ".join(f"{rel}: {h}" for rel, h in sorted(hosts.items())))


def test_the_default_is_reachable_without_configuration():
    """The point of the default: a reader with an OPENAI_API_KEY and nothing else works.

    Without this, every check above would still pass on an all-gateway tree -- which is
    what this repository shipped before, and is the state this guard exists to prevent
    returning to.
    """
    hosts = {urlparse(_defaults(rel)[0]).netloc
             for rel in SITES if SITES[rel][0] is not None}
    assert hosts == {OPENAI_HOST}, (
        f"the judge defaults point at {sorted(hosts)}, which a reader outside the "
        f"organisation cannot reach. docs/install.md section 4 is how they opt back in.")
