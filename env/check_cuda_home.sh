#!/bin/bash
# Validate CUDA_HOME on the *execution* node, before launching training.
#
# The GRPO / cold-start launchers run DeepSpeed, whose op builder JIT-compiles
# fused ops and needs a real CUDA toolkit (nvcc) at $CUDA_HOME. A stale or
# hardcoded CUDA_HOME (e.g. a toolkit path copied from a different cluster, like
# /cm/shared/apps/cuda12.4/... which does not exist on every cluster) otherwise
# fails deep inside DeepSpeed at the first op build with a cryptic error. This
# check fails fast with an actionable message instead.
#
# Usage (from a launch script, on the node):  bash "$REPO/check_cuda_home.sh" || exit 1
set -u

fail() {
    {
        echo "========================================================================"
        echo "ERROR: invalid CUDA_HOME on execution node $(hostname)"
        echo "  CUDA_HOME='${CUDA_HOME:-<unset>}'"
        echo "  $1"
        echo "  DeepSpeed's op builder needs nvcc to JIT-compile fused ops -> training would fail."
        echo ""
        echo "  Fix: re-submit with CUDA_HOME pointing at a real toolkit on the compute node:"
        echo "      CUDA_HOME=/path/to/cuda bash <launch_script> ..."
        echo "  Find a toolkit on THIS node:"
        echo "      ls -d /usr/local/cuda* /cm/shared/apps/cuda*/toolkit/* 2>/dev/null"
        echo "      command -v nvcc"
        echo "      module avail cuda 2>&1 | head"
        echo "========================================================================"
    } >&2
    exit 1
}

if [ "${DS_BUILD_OPS:-0}" = "0" ]; then
    echo "CUDA_HOME check skipped on $(hostname): DS_BUILD_OPS=0, no JIT compilation will occur."
    exit 0
fi

[ -n "${CUDA_HOME:-}" ]      || fail "CUDA_HOME is empty/unset."
[ -d "$CUDA_HOME" ]          || fail "CUDA_HOME directory does not exist."
[ -x "$CUDA_HOME/bin/nvcc" ] || fail "No executable nvcc at \$CUDA_HOME/bin/nvcc."

echo "CUDA_HOME OK on $(hostname): $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1))"
