# Copyright 2026 NVIDIA. Apache-2.0.
"""A stand-in for a research module that stayed in the archive.

Several probes here were written against saliency maps and controls that were explored
and dropped: the pixel-gradient map, the GLIMPSE map, the attention-rollout flow, and the
roll-null, placebo, mask-free, mismatched-box and length-guard rewards. None of them has
a number in the paper, so none of them came across; `docs/provenance.md` lists them and
they are whole and still runnable in the archive repo.

The probes themselves DID come across, because their attention-map path is the one the
paper uses. So each names its missing dependency as one of these, and the map or stage
that needed it fails when it is USED -- with a sentence saying where the code went --
instead of at an import three screens earlier with a traceback about a path that means
nothing to the reader.

    GM = unavailable("grad", "trl/grad_maps.py", "trl/rewards/grad_rewards.py")

NOTHING MAY READ AN ATTRIBUTE OFF ONE OF THESE AT IMPORT OR PARSER-BUILD TIME. That is a
sharper rule than it looks, and it is the rule this pattern was breaking everywhere it was
used. Two ways to break it, neither of which looks like an attribute read:

    def f(..., chunk=GM.SPAN_CHUNK_DEFAULT)      evaluated when the MODULE is imported
    p.add_argument(..., default=GM.X)            evaluated while the PARSER is built

Either one turns "fails at the flag" into "fails on import", which is how
`experiments/trained_model/probe.py` came to be unimportable -- so `--map attn`, the only
map in the paper, was as dead as the two that are not here. Write the literal out instead.
`tests/test_dropped_variants_fail_late.py` enforces this.
"""

from __future__ import annotations

#: The archive repository, as it is referred to in every one of these messages.
ARCHIVE = "research/saliency_r1"


class Unavailable:
    """Raises NotImplementedError on any attribute access, naming what is missing."""

    def __init__(self, name: str, *paths: str):
        self._name = name
        self._paths = paths

    def __repr__(self) -> str:
        return f"<unavailable: {self._name}>"

    def __getattr__(self, attr):
        # Dunders must still raise AttributeError rather than NotImplementedError:
        # `inspect`, `copy` and pytest's assertion rewriting all probe an object for
        # __class__, __name__ and friends while formatting a traceback, and answering
        # those with the wrong exception type turns a clear failure into a confusing one.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        where = ", ".join(self._paths)
        raise NotImplementedError(
            f"`{self._name}` is not part of this repository, and neither is anything "
            f"that depends on it. No arm or figure in the paper was produced with it "
            f"(see docs/provenance.md). It is intact in the archive repo "
            f"({ARCHIVE}: {where}). Tried to reach `{self._name}.{attr}`.")


def unavailable(name: str, *paths: str) -> Unavailable:
    """`Unavailable(name, *paths)`, spelled as a call at the use site."""
    return Unavailable(name, *paths)
