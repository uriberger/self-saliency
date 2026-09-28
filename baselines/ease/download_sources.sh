#!/usr/bin/env bash
#SBATCH --job-name=ease-dl
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/ease_download_%j.log
#
# Download the public source corpora behind the EASE evidence pools
# (arXiv:2605.30912, Appendix B.4 Table 5). The paper names its five sources and
# their per-source counts but never released the box annotations, so we pull the
# sources and re-run its three-step pipeline ourselves.
#
#   ZwZ74K      74,000 single / 0 multi      inclusionAI/ZwZ-RL-VQA    Apache-2.0
#   ViRL39K     28,886 single / 9,978 multi  TIGER-Lab/ViRL39K         MIT
#   SpaCE10      2,371 single / 1,759 multi  Cusyoung/SpaCE-10         MIT
#   CLEVR        1,531 single / 4,528 multi  handled separately (gold scene graphs)
#   SuperCLEVR     926 single / 3,992 multi  handled separately (gold scene graphs)
#
# ZwZ ships TWO image trees. `images/` (178 GB) has the evidence boxes BURNED IN
# -- that is how ZwZ itself trains. EASE's convention is the opposite: "evidence
# boxes are never rendered into the image", and prepare_ease_dataset.py actively
# rejects paths matching with_boxes/marked/annotated/visualized. So we take
# `original_images/` (48 GB) plus the parquet `bbox` column as metadata, and skip
# `images/` entirely. That is a 4.7x smaller download AND the correct one.
#
# Usage:
#   sbatch download_ease_sources.sh                 # all corpora
#   bash   download_ease_sources.sh --only zwz      # one corpus, here on the login node
#   bash   download_ease_sources.sh --dest /path    # override destination
#
# Idempotent: hf download resumes, and extraction is skipped when the marker
# file for that corpus already exists.

set -euo pipefail

# Under sbatch the script is copied into the Slurm spool dir, so BASH_SOURCE
# does NOT point at the repo. SLURM_SUBMIT_DIR is the only reliable anchor.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
DEST="$REPO/cold_data/ease"
PYBIN=/home/uberger/scratch/miniconda3/envs/saliency_r1_qwen3/bin
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$REPO/logs" "$DEST/raw" "$DEST/images"
export PATH="$PYBIN:$PATH"
export HF_HUB_DISABLE_TELEMETRY=1

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

say() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- ZwZ74K -----
if want zwz; then
    say "ZwZ-RL-VQA: parquets + original_images (48 GB; skipping the 178 GB box-burned images/)"
    # NB: --include must be REPEATED. Space-separated patterns after a single
    # --include are parsed as positional filenames, and the CLI then warns
    # "Ignoring --include since filenames have been explicitly set" and silently
    # downloads only the first one -- which would skip all 48 GB of images.
    hf download inclusionAI/ZwZ-RL-VQA \
        --repo-type dataset \
        --local-dir "$DEST/raw/zwz" \
        --include "train.parquet" \
        --include "train1.parquet" \
        --include "original_images/*" \
        --include "README.md"

    if [[ ! -f "$DEST/images/zwz/.extracted" ]]; then
        say "ZwZ: merging 5 tarball parts and extracting"
        mkdir -p "$DEST/images/zwz"
        cat "$DEST"/raw/zwz/original_images/original_images.tar.gz* \
            | tar -xf - -C "$DEST/images/zwz"
        touch "$DEST/images/zwz/.extracted"
    else
        say "ZwZ: already extracted, skipping"
    fi
fi

# --------------------------------------------------------------- ViRL39K -----
if want virl; then
    say "ViRL39K: full repo (1.8 GB)"
    hf download TIGER-Lab/ViRL39K \
        --repo-type dataset \
        --local-dir "$DEST/raw/virl39k"

    if [[ ! -f "$DEST/images/virl39k/.extracted" ]]; then
        say "ViRL39K: unzipping images"
        mkdir -p "$DEST/images/virl39k"
        unzip -q -o "$DEST/raw/virl39k/images.zip" -d "$DEST/images/virl39k"
        touch "$DEST/images/virl39k/.extracted"
    else
        say "ViRL39K: already extracted, skipping"
    fi
fi

# --------------------------------------------------------------- SpaCE-10 ----
if want space; then
    # NOTE: SpaCE-10 publishes a single `test` split of 4,132 rows. EASE reports
    # 4,130 SpaCE10 training examples, i.e. it trained on essentially the whole
    # benchmark. We reproduce that, but it is worth knowing the provenance.
    say "SpaCE-10: full repo (0.9 GB, test split only -- see note above)"
    hf download Cusyoung/SpaCE-10 \
        --repo-type dataset \
        --local-dir "$DEST/raw/space10"
fi

say "done"
du -sh "$DEST"/raw/* "$DEST"/images/* 2>/dev/null || true
