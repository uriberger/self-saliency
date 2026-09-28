#!/usr/bin/env bash
# Apply our changes to ease_repo/ (their EasyR1 fork, gitignored and shared).
#
# TWO changes, and neither is in the EASE method:
#
#   verl/workers/reward/function.py
#     -- Add `question` and `data_source` to the dicts handed to the reward
#        function. Their AutoRewardManager passes only {response,
#        response_length, ground_truth}, which is all a rule matcher needs; an
#        LLM-judge reward cannot grade an answer it cannot see the question for.
#        Purely additive: extra dict keys are invisible to their own
#        examples/reward_function/perception.py, so the DAPO baseline arm runs
#        on unmodified behaviour.
#
#   verl/utils/vllm_utils.py
#     -- `from vllm.lora.lora_model import LoRAModel` does not resolve under
#        vllm 0.11.0, which is the version their own Dockerfile pins: 0.11.0
#        keeps LoRAModel in vllm.lora.models and only a later refactor split it
#        out. The import is unconditional and sits under
#        verl/workers/rollout/__init__.py, so the trainer dies on `import verl`
#        -- before any config is read, and whether or not LoRA is used (it is
#        not; config.yaml sets lora.rank 0). Replaced with a try/except that
#        accepts either module path.
#
# Nothing under verl/workers/actor/ is touched -- evidence_mask.py,
# trainable_attention.py and dp_actor.py are the EASE method itself and stay
# byte-identical, which is the whole point of running their framework.
#
# Idempotent: the edit is guarded by a grep, and the original is kept as
# function.py.orig on first run.
#
# Usage:
#   bash patch_ease_repo.sh
#   bash patch_ease_repo.sh --repo /path/to/ease_repo
#   bash patch_ease_repo.sh --revert
#
# ease_repo/ is symlinked into every worktree, so this patches the copy every
# session and every running job uses. It is additive and safe to re-run, but it
# is not worktree-local.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
EASE_REPO="$REPO/ease_repo"
REVERT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) EASE_REPO="$2"; shift 2 ;;
        --revert) REVERT=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

TARGET="$EASE_REPO/verl/workers/reward/function.py"
VLLM_UTILS="$EASE_REPO/verl/utils/vllm_utils.py"
for f in "$TARGET" "$VLLM_UTILS"; do
    [ -f "$f" ] || { echo "MISSING: $f -- is ease_repo/ cloned?" >&2; exit 1; }
done

echo "=== patch_ease_repo.sh: ease_repo=$EASE_REPO ==="

if [[ "$REVERT" == 1 ]]; then
    for f in "$TARGET" "$VLLM_UTILS"; do
        if [ -f "$f.orig" ]; then
            mv "$f.orig" "$f"
            echo "  reverted $f"
        else
            echo "  nothing to revert for $f"
        fi
    done
    exit 0
fi

# ── 2. vllm_utils.py: LoRAModel moved between vllm versions ─────────────────
if grep -q 'vllm.lora.models import LoRAModel' "$VLLM_UTILS"; then
    echo "  (vllm_utils.py already patched - skipping)"
else
    [ -f "$VLLM_UTILS.orig" ] || cp "$VLLM_UTILS" "$VLLM_UTILS.orig"
    python3 - "$VLLM_UTILS" <<'PYEOF2'
import sys

path = sys.argv[1]
src = open(path).read()

OLD = "from vllm.lora.lora_model import LoRAModel\n"
NEW = """try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:  # vllm <= 0.11.x keeps LoRAModel in vllm.lora.models
    from vllm.lora.models import LoRAModel
"""
if OLD not in src:
    sys.exit(f"anchor not found in {path}: {OLD!r}")
open(path, "w").write(src.replace(OLD, NEW, 1))
print(f"  patched {path}")
PYEOF2
    python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$VLLM_UTILS"
fi

# ── 1. function.py: widen the reward-function interface ─────────────────────
if grep -q '_extra_reward_field' "$TARGET"; then
    echo "  (function.py already patched - skipping)"
    echo "=== done ==="
    exit 0
fi

[ -f "$TARGET.orig" ] || cp "$TARGET" "$TARGET.orig"

python3 - "$TARGET" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

HELPER = '''

def _extra_reward_field(data: "DataProto", index: int, key: str) -> str:
    """Best-effort lookup of a per-sample string field for the reward function.

    Added by saliency_r1's patch_ease_repo.sh. EASE's own perception reward is
    rule-based and needs nothing beyond the gold string, but an LLM-judge reward
    has to see the question that was asked. `question` lives inside the
    `extra_info` dict that scripts/prepare_ease_dataset.py writes; `data_source`
    is a top-level column. Datasets carrying neither get "", so this is safe for
    any parquet and invisible to reward functions that ignore the keys.
    """
    non_tensor = data.non_tensor_batch
    if key in non_tensor:
        value = non_tensor[key][index]
        return "" if value is None else str(value)

    extra_info = non_tensor.get("extra_info")
    if extra_info is not None:
        info = extra_info[index]
        if isinstance(info, dict) and info.get(key) is not None:
            return str(info[key])

    return ""

'''

ANCHOR = "class SequentialFunctionRewardManagerMixin:"
if ANCHOR not in src:
    sys.exit(f"anchor not found in {path}: {ANCHOR!r}")
src = src.replace(ANCHOR, HELPER.lstrip("\n") + "\n" + ANCHOR, 1)

OLD = '''                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
'''
NEW = '''                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "question": _extra_reward_field(data, i, "question"),
                    "data_source": _extra_reward_field(data, i, "data_source"),
'''
count = src.count(OLD)
if count != 2:
    sys.exit(f"expected 2 reward-input dicts in {path}, found {count}")
src = src.replace(OLD, NEW)

# Keep the TypedDict honest about what a reward function now receives.
src = src.replace(
    '''class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str
''',
    '''class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str
    question: str  # added by saliency_r1's patch_ease_repo.sh
    data_source: str  # added by saliency_r1's patch_ease_repo.sh
''',
    1,
)

open(path, "w").write(src)
print(f"  patched {path}")
PYEOF

python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$TARGET"
echo "  syntax OK"
echo "=== done ==="
