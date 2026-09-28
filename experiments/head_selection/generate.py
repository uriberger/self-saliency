"""Run a reasoning VLM on a visual reasoning benchmark and record per-sample
reasoning chains plus correctness.

Defaults
--------
  model:   Qwen/Qwen3-VL-8B-Thinking
  dataset: nirajandhakal/realworldqa  (split=test)

Efficiency methods applied
--------------------------
  * Flash Attention 2  (attn_implementation="flash_attention_2"),
    with a graceful fallback to PyTorch SDPA if flash-attn is not installed.
  * bfloat16 weights   (torch_dtype=torch.bfloat16)
  * device_map="auto"  for automatic GPU placement
  * KV cache enabled   (use_cache=True; transformers default, kept explicit)
  * torch.inference_mode() around generation (no autograd bookkeeping)
  * Single-sample inference (batch size 1): reasoning chains vary by an order
    of magnitude in length, so padded batched generation wastes VRAM and
    rarely improves wall-clock on long-form decoding.
  * The model's own recommended generation_config (sampling params shipped
    with the checkpoint) is used as-is; only max_new_tokens is overridden.

Output
------
A JSONL file (default: results.jsonl) with one entry per sample:
  - sample_id        dataset id (or row index if absent)
  - image_filename   dataset row index (per spec; images not saved to disk)
  - question
  - ground_truth
  - response         full raw model output
  - reasoning        text inside <think>...</think> (if present)
  - final_answer     text after </think> (or full response if no tags)
  - correct          for single-letter (multiple-choice) ground truths, whether
                     the option the model committed to (extract_mcq_choice)
                     equals the GT letter; otherwise a case-insensitive
                     substring match between final_answer and ground_truth
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import string
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, set_seed
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
except ImportError:
    # Allow --merge to run without the full ML stack installed.
    np = torch = T = InterpolationMode = load_dataset = AutoModel = None
    AutoModelForCausalLM = AutoProcessor = set_seed = AutoModelForImageTextToText = None


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_DATASET = "nirajandhakal/realworldqa"
DEFAULT_SPLIT = "test"

# Dataset registry (slug → (hf_id, config_or_none, split))
# config_or_none is the HF dataset config name (second positional arg to load_dataset).
_DATASET_REGISTRY: dict = {
    "vstar":                 ("craigwu/vstar_bench",          None, "test"),
    "vstar_direct":          ("craigwu/vstar_bench",          None, "test"),
    "vstar_position":        ("craigwu/vstar_bench",          None, "test"),
    "hrbench":               ("DreamMr/HR-Bench",             None, "test"),
    "realworldqa":           ("nirajandhakal/realworldqa",    None, "test"),
    "realworldqa_truncated": ("nirajandhakal/realworldqa",    None, "test"),
    "mathvista":             ("AI4Math/MathVista",            None, "testmini"),
    "algopuzzlevqa":         ("declare-lab/AlgoPuzzleVQA",   None, "data"),
    "mmmu_pro_vision":       ("MMMU/MMMU_Pro",               "vision", "test"),
    "spatialqa":             ("xyc99/SpatiaLQA",             None, "train"),
    "visulogic":             None,  # special loader: see _load_visulogic()
    "crossmath":             None,  # special loader: see _load_crossmath()
    "saliency_r1_8k":        ("peterant330/saliency-r1-8k",   None, "train"),
    "dailyclue":             None,  # special loader: see _load_dailyclue()
    "omnispatial":           None,  # special loader: see _load_omnispatial()
    "virl39k":               None,  # special loader: see _load_virl39k()
    "visual_cot":            None,  # special loader: see _load_visual_cot()
}

# Datasets whose parquet files contain invalid ClassLabel values (e.g. a git commit
# hash stored in a ClassLabel column).  These must be loaded via streaming so the
# broken encode step is bypassed; ClassLabel columns are then cast to plain strings
# and the result is materialised into a regular Dataset.
_STREAMING_CAST_DATASETS: set = {"xyc99/SpatiaLQA"}


def _load_visulogic():
    """Load the VisuLogic benchmark.

    The HuggingFace dataset VisuLogic/VisuLogic only exposes images in its
    parquet; questions and labels live in a separate data.jsonl file in the
    same repo.  This loader downloads both files, extracts images.zip to a
    local cache directory, and returns a Dataset with the fields the main
    inference loop expects: image (PIL), question, label, tag, id.
    """
    import zipfile
    from io import BytesIO
    from datasets import Dataset as _Dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image as _PILImage

    jsonl_path = hf_hub_download(
        repo_id="VisuLogic/VisuLogic", filename="data.jsonl", repo_type="dataset"
    )
    zip_path = hf_hub_download(
        repo_id="VisuLogic/VisuLogic", filename="images.zip", repo_type="dataset"
    )

    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with zipfile.ZipFile(zip_path) as z:
        raw_images: dict = {}
        for name in z.namelist():
            if name.lower().endswith(".png") or name.lower().endswith(".jpg"):
                with z.open(name) as img_f:
                    raw_images[name] = img_f.read()

    dataset_rows = []
    for row in rows:
        img_key = row["image_path"]  # e.g. 'images/00000.png'
        if img_key not in raw_images:
            continue
        img = _PILImage.open(BytesIO(raw_images[img_key])).convert("RGB").copy()
        dataset_rows.append({
            "image": img,
            "question": row["question"],
            "label": row["label"],
            "tag": row["tag"],
            "id": row["id"],
        })

    return _Dataset.from_list(dataset_rows)


# Module-level state for lazy, per-image extraction from ViRL39K's images.zip.
_VIRL39K_ZIP = None       # cached open zipfile.ZipFile handle
_VIRL39K_IMG_ROOT = None  # local extraction cache dir


def _virl39k_ensure_image(abs_path):
    """Extract a single ViRL39K image from images.zip on first access, then cache.

    ViRL39K ships all ~39k images in one 1.77 GB zip. Extracting them all up front
    is very slow on a parallel filesystem (many-small-file creates), so we extract
    each image lazily the first time it is actually requested and leave it on disk
    for subsequent runs. Called from _resolve_image for paths under the cache dir.
    """
    import os
    if _VIRL39K_IMG_ROOT is None or _VIRL39K_ZIP is None or os.path.exists(abs_path):
        return abs_path
    rel = os.path.relpath(abs_path, _VIRL39K_IMG_ROOT)  # e.g. images/xxxx.jpg
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    try:
        with _VIRL39K_ZIP.open(rel) as src, open(abs_path, "wb") as dst:
            dst.write(src.read())
    except KeyError:
        # Fallback: some zips flatten the directory structure.
        with _VIRL39K_ZIP.open(os.path.basename(rel)) as src, open(abs_path, "wb") as dst:
            dst.write(src.read())
    return abs_path


def _load_virl39k(split_override=None):
    """Load TIGER-Lab/ViRL39K, a VLM-reasoning RL training set (~38.8K QA pairs).

    The HF repo ships QA rows in ``39Krelease.parquet`` (with per-row image *paths*)
    and all images bundled in a single ``images.zip`` (~1.77 GB) — the images are
    NOT individually addressable in the repo. Rather than extract all ~39k images
    up front (slow on lustre), this loader downloads the zip once and extracts each
    image *lazily* on first access (see _virl39k_ensure_image, invoked from
    _resolve_image). It returns a Dataset with the fields the inference loop and
    aggregation_correlation collect expect:
      - image    : absolute path string to the (lazily) extracted jpg
      - question : question text with the ``<image>`` placeholder stripped
      - answer   : ground-truth answer with ``\\boxed{...}`` unwrapped
      - qid, source, category : passthrough metadata

    Note: images are returned as a *scalar* path (not a 1-element list) so both the
    inference loop and collect resolve them identically via _resolve_image.
    """
    import os
    import re as _re
    import zipfile
    import pandas as pd
    from datasets import Dataset as _Dataset
    from huggingface_hub import hf_hub_download

    global _VIRL39K_ZIP, _VIRL39K_IMG_ROOT

    repo = "TIGER-Lab/ViRL39K"
    parquet_path = hf_hub_download(
        repo_id=repo, filename="39Krelease.parquet", repo_type="dataset"
    )
    zip_path = hf_hub_download(repo_id=repo, filename="images.zip", repo_type="dataset")

    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    _VIRL39K_IMG_ROOT = os.path.join(cache_root, "virl39k_images")
    os.makedirs(_VIRL39K_IMG_ROOT, exist_ok=True)
    _VIRL39K_ZIP = zipfile.ZipFile(zip_path)  # images extracted lazily on demand

    def _clean_q(q):
        return _re.sub(r"<image>", "", str(q)).strip()

    def _clean_a(a):
        m = _re.search(r"\\boxed\{(.*)\}", str(a))
        return (m.group(1) if m else str(a)).strip()

    df = pd.read_parquet(parquet_path)
    rows = []
    for row in df.to_dict("records"):
        imgs = row.get("image")
        if imgs is None or len(imgs) == 0:
            continue
        rows.append({
            "image":    os.path.join(_VIRL39K_IMG_ROOT, imgs[0]),
            "question": _clean_q(row.get("question", "")),
            "answer":   _clean_a(row.get("answer", "")),
            "qid":      row.get("qid", ""),
            "source":   row.get("source", ""),
            "category": row.get("category", ""),
        })
    return _Dataset.from_list(rows)


# --- Visual-CoT (deepcs233/Visual-CoT) -----------------------------------------
# Real-world, region-grounded VQA across 12 sources (GQA/COCO/VG, TextVQA/DocVQA/
# OCR, CUB, Flickr30k/OpenImages, VSR, …); the superset saliency_r1_8k was drawn
# from. Each row ships a human key-region bounding box (avoids the Grounding-DINO
# quality problem seen on virl39k). ~434k boxed QA rows across 12 metadata JSONLs;
# all images bundled in a 13-part split tar (~139 GB) that concatenates into one
# tar. Images are extracted lazily on first access (like the ViRL39K zip loader).
_VISCOT_REPO = "deepcs233/Visual-CoT"
_VISCOT_SOURCES = ["cub", "docvqa", "dude", "flickr30k", "gqa", "infographicsvqa",
                   "openimages", "sroie", "textcap", "textvqa", "visual7w", "vsr"]
_VISCOT_IMG_ROOT = None       # cache dir for lazily-extracted images
_VISCOT_TARFILE = None        # open tarfile.TarFile over the combined image tar
_VISCOT_MEMBER_BY_BASE = None # basename -> [member_name, ...]


def _viscot_combined_tar():
    """Concatenate the 13 pre-downloaded image-tar parts into a single
    ``cot_images.tar`` (once). Returns the combined tar path.

    Parts are fetched separately by ``scripts/viscot_download.py`` (a robust
    direct-HTTP resumable downloader) into ``$HF_HOME/viscot_images/parts/`` — run
    that first (it is idempotent). We deliberately do NOT download here so the
    139 GB pull never happens inside a GPU inference job."""
    import os
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    combined = os.path.join(cache_root, "viscot_images", "cot_images.tar")
    os.makedirs(os.path.dirname(combined), exist_ok=True)
    if os.path.exists(combined) and os.path.getsize(combined) > 130e9:
        return combined
    parts_dir = os.path.join(cache_root, "viscot_images", "parts")
    parts = [os.path.join(parts_dir, f"cot_images_{i:02d}") for i in range(13)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"[visual_cot] {len(missing)} image-tar parts missing from {parts_dir}. "
            f"Run: ./venv/bin/python scripts/viscot_download.py  (resumable, ~139 GB) "
            f"before running inference.")
    tmp = combined + ".tmp"
    log(f"[visual_cot] concatenating 13 image-tar parts -> {combined} (one-time)")
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 24)
                    if not chunk:
                        break
                    out.write(chunk)
    os.replace(tmp, combined)
    return combined


def _viscot_open_tar():
    """Open the combined tar and build/reuse a cached basename->member index."""
    global _VISCOT_TARFILE, _VISCOT_MEMBER_BY_BASE
    if _VISCOT_TARFILE is not None:
        return
    import os, tarfile, pickle
    combined = _viscot_combined_tar()
    idx_path = combined + ".baseidx.pkl"
    _VISCOT_TARFILE = tarfile.open(combined, "r:")
    if os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            _VISCOT_MEMBER_BY_BASE = pickle.load(f)
        return
    log("[visual_cot] indexing image tar (one-time, scans member headers)...")
    base_map: dict = {}
    for m in _VISCOT_TARFILE:            # streaming: reads headers, seeks past data
        if m.isfile():
            base_map.setdefault(os.path.basename(m.name), []).append(m.name)
    _VISCOT_MEMBER_BY_BASE = base_map
    with open(idx_path, "wb") as f:
        pickle.dump(base_map, f)
    log(f"[visual_cot] indexed {len(base_map)} image basenames")


def _viscot_ensure_image(abs_path):
    """Extract one Visual-CoT image from the combined tar on first access, cache it.
    ``abs_path`` is ``<IMG_ROOT>/<dataset>/<basename>``."""
    import os
    if _VISCOT_IMG_ROOT is None or os.path.exists(abs_path):
        return abs_path
    _viscot_open_tar()
    dataset = os.path.relpath(abs_path, _VISCOT_IMG_ROOT).split(os.sep)[0]
    base = os.path.basename(abs_path)
    candidates = _VISCOT_MEMBER_BY_BASE.get(base, [])
    if not candidates:
        raise KeyError(f"[visual_cot] image not found in tar: {base}")
    member = candidates[0]
    if len(candidates) > 1:  # disambiguate collisions by the dataset hint in the path
        pref = [c for c in candidates if dataset.lower() in c.lower()]
        member = pref[0] if pref else candidates[0]
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    src = _VISCOT_TARFILE.extractfile(member)
    tmp = abs_path + ".tmp"
    with open(tmp, "wb") as dst:
        dst.write(src.read())
    os.replace(tmp, abs_path)
    return abs_path


def _load_visual_cot(sample_n=None, seed=None):
    """Load a deterministic random subset of Visual-CoT, excluding rows already in
    saliency_r1_8k so it is an independent set.

    Env overrides (keep consistent across resumes so the subset is identical):
      VISCOT_SAMPLE_N (default 20000), VISCOT_SEED (default 0).

    Returns a Dataset with: image (lazy-extract path), question, answer, id
    (stable ``dataset:basename:qhash`` for resume), bbox (JSON of normalized
    [x1,y1,x2,y2] human key-region boxes), dataset.
    """
    import os, json, hashlib
    import numpy as np
    import pandas as pd
    from datasets import Dataset as _Dataset
    from huggingface_hub import hf_hub_download

    global _VISCOT_IMG_ROOT
    sample_n = int(os.environ.get("VISCOT_SAMPLE_N", sample_n if sample_n is not None else 20000))
    seed = int(os.environ.get("VISCOT_SEED", seed if seed is not None else 0))
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    _VISCOT_IMG_ROOT = os.path.join(cache_root, "viscot_images", "extracted")
    os.makedirs(_VISCOT_IMG_ROOT, exist_ok=True)

    # Instance-level exclusion of saliency_r1_8k rows. saliency_r1_8k ships no image
    # filename, but its normalized key-region bbox matches Visual-CoT's to ~1e-3, so
    # (question, answer) + bbox proximity uniquely identifies the ~8,080 instances.
    # A plain (dataset, question) key would over-drop templated questions (e.g. CUB
    # "Does the bird have ...?" yes/no) that recur across thousands of other images.
    BBOX_TOL = 0.02
    sal_bbox: dict = {}   # (question, answer) -> [normalized [x1,y1,x2,y2], ...]
    for i in range(3):
        pp = hf_hub_download("peterant330/saliency-r1-8k",
                             f"data/train-{i:05d}-of-00003.parquet", repo_type="dataset")
        sdf = pd.read_parquet(pp, columns=["problem", "solution", "bbox"])
        for q, a, bb in zip(sdf["problem"], sdf["solution"], sdf["bbox"]):
            sal_bbox.setdefault((str(q).strip(), str(a).strip()), []).append(json.loads(bb))

    def _norm_boxes(r):
        """Normalize + clip Visual-CoT pixel bboxs to [x1,y1,x2,y2] in [0,1]."""
        W, H = r.get("width") or 0, r.get("height") or 0
        out = []
        for b in (r.get("bboxs") or []):
            if len(b) >= 4 and W and H:
                out.append([min(1.0, max(0.0, b[0] / W)), min(1.0, max(0.0, b[1] / H)),
                            min(1.0, max(0.0, b[2] / W)), min(1.0, max(0.0, b[3] / H))])
        return out

    def _is_saliency_row(q, a, nb):
        cands = sal_bbox.get((q, a))
        if not cands or nb is None:
            return False
        return any(max(abs(nb[k] - c[k]) for k in range(4)) <= BBOX_TOL for c in cands)

    # Load all source metadata rows, dropping the saliency_r1_8k instances.
    rows, n_excl = [], 0
    for src in _VISCOT_SOURCES:
        p = hf_hub_download(_VISCOT_REPO, f"metadata/{src}_cot_train.jsonl", repo_type="dataset")
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                r.setdefault("dataset", src)
                boxes = _norm_boxes(r)
                q = str(r.get("question", "")).strip()
                a = str(r.get("answer", "")).strip()
                if _is_saliency_row(q, a, tuple(boxes[0]) if boxes else None):
                    n_excl += 1
                    continue
                r["_norm_boxes"] = boxes
                rows.append(r)
    log(f"[visual_cot] {len(rows)} rows after excluding {n_excl} saliency_r1_8k instances")

    # Deterministic random subsample (stable across resumes for a fixed seed).
    if sample_n and sample_n < len(rows):
        idx = np.random.default_rng(seed).permutation(len(rows))[:sample_n]
        rows = [rows[i] for i in sorted(idx.tolist())]
    log(f"[visual_cot] sampled {len(rows)} rows (seed={seed})")

    out = []
    for r in rows:
        ds = str(r.get("dataset", ""))
        img = str(r.get("image", ""))
        q = str(r.get("question", "")).strip()
        boxes = r.get("_norm_boxes", [])
        base = os.path.basename(img)
        a = str(r.get("answer", "")).strip()
        # content hash (q + a + box) so distinct rows on the same image get distinct
        # ids; only exact-duplicate rows collide, which resume can safely skip.
        _h = hashlib.md5(f"{q}|{a}|{json.dumps(boxes)}".encode("utf-8")).hexdigest()[:10]
        qid = f"{ds}:{base}:{_h}"
        out.append({
            "image":    os.path.join(_VISCOT_IMG_ROOT, ds, base),
            "question": q,
            "answer":   str(r.get("answer", "")).strip(),
            "id":       qid,
            "bbox":     json.dumps(boxes),
            "dataset":  ds,
        })
    return _Dataset.from_list(out)


def _load_crossmath(config=None):
    """Load the CrossMath benchmark.

    CrossMath has no 'question' field; the puzzle layout is in 'markdown_table'
    and the answer is in 'solutions'.  This loader adds synthesized 'question'
    and 'ground_truth' fields so the standard inference loop works unchanged.

    The dataset has four style configs: Original (default), Altstyle, Beige,
    Noborder.  Pass --subset <StyleName> to select a non-default style.
    """
    style = config or "Original"
    ds = load_dataset("xuyige/CrossMath", style, split="test")

    def _remap(row):
        markdown_table = row.get("markdown_table", "")
        question = (
            "Solve the math crossword puzzle shown in the image. "
            "Each row and column must satisfy the given equation. "
            "Find all missing values (marked with ?).\n\n"
            + markdown_table
        )
        return {"question": question, "ground_truth": row.get("solutions", "")}

    return ds.map(_remap)


def _load_dailyclue():
    """Load the DailyClue benchmark.

    Crysun/DailyClue's default HF parquet view (what plain load_dataset()
    returns) is an artifact of the hub's auto dataset-viewer conversion: it
    only exposes 'image' + a coarse 4-way 'label' (one of the domain folder
    names). The actual benchmark data - question, ground_truth, answer
    format, and human-annotated clues - lives in four per-domain JSON files
    (daily_life.json, location.json, science.json, spatial.json) plus loose
    image files in the matching folders, none of which are surfaced by the
    default load_dataset() path. This loader downloads those JSON files and
    their referenced images directly and reconstructs the real VQA schema.
    """
    import json as _json
    from datasets import Dataset as _Dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image as _PILImage

    domains = ["daily_life", "location", "science", "spatial"]
    rows = []
    for domain in domains:
        json_path = hf_hub_download(
            repo_id="Crysun/DailyClue", filename=f"{domain}.json", repo_type="dataset"
        )
        with open(json_path) as f:
            items = _json.load(f)
        for item in items:
            image_name = item["image"][0]
            image_path = hf_hub_download(
                repo_id="Crysun/DailyClue",
                filename=f"{domain}/{image_name}",
                repo_type="dataset",
            )
            image = _PILImage.open(image_path).convert("RGB").copy()
            rows.append({
                "image": image,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "clues": item.get("clues", ""),
                "format": item.get("format", ""),
                "category_1": item.get("category_1", ""),
                "category_2": item.get("category_2", ""),
            })

    return _Dataset.from_list(rows)


_OMNISPATIAL_ZIP_BY_SPLIT = {
    "test": "OmniSpatial-test.zip",
    "train": "OmniSpatial-train.zip",
    "full": "OmniSpatial-full.zip",
}


def _load_omnispatial(split_override=None):
    """Load the OmniSpatial benchmark.

    qizekun/OmniSpatial ships no loading script and no top-level parquet/csv/json
    - just per-split zip files (OmniSpatial-{test,train,full}.zip). Plain
    load_dataset() falls back to HF's auto parquet-conversion, which is broken for
    this repo (it corrupts a ClassLabel feature with a leaked repo@revision
    string). This loader downloads the requested split's zip directly and reads
    data.json plus the referenced per-category images out of it in memory.

    Each zip unpacks to {root}/data.json + {root}/{task_type}/{image_num}.png,
    where an item's "id" field ({image_num}_{question_num}) links a question to
    its image. "answer" is an index into "options"; it is converted to the
    matching letter (A, B, ...) so it lines up with the lettered choices the main
    loop appends to the question text.
    """
    import zipfile
    from io import BytesIO
    from datasets import Dataset as _Dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image as _PILImage

    split = split_override or "test"
    zip_name = _OMNISPATIAL_ZIP_BY_SPLIT.get(split)
    if zip_name is None:
        raise ValueError(
            f"Unknown OmniSpatial split '{split}'; expected one of "
            f"{sorted(_OMNISPATIAL_ZIP_BY_SPLIT)}."
        )
    zip_path = hf_hub_download(repo_id="qizekun/OmniSpatial", filename=zip_name, repo_type="dataset")
    root_dir = zip_name[: -len(".zip")]

    rows = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{root_dir}/data.json") as f:
            items = json.load(f)
        for item in items:
            img_num = item["id"].split("_")[0]
            img_key = f"{root_dir}/{item['task_type']}/{img_num}.png"
            with z.open(img_key) as img_f:
                img = _PILImage.open(BytesIO(img_f.read())).convert("RGB").copy()
            answer_idx = item["answer"]
            ground_truth = (
                string.ascii_uppercase[answer_idx]
                if isinstance(answer_idx, int)
                else answer_idx
            )
            rows.append({
                "image": img,
                "question": item["question"],
                "options": item["options"],
                "ground_truth": ground_truth,
                "task_type": item["task_type"],
                "sub_task_type": item.get("sub_task_type", ""),
                "id": f"{item['task_type']}_{item['id']}",
            })

    return _Dataset.from_list(rows)


def _load_streaming_cast(hf_id, config, split):
    """Load a dataset that has broken ClassLabel features.

    Uses streaming=True to bypass the encode-time ClassLabel validation, casts
    every ClassLabel column to a plain string, then materialises the result into
    a regular (non-iterable) Dataset so the rest of the pipeline works unchanged.
    """
    from datasets import Value, Dataset as _Dataset
    load_kw = {"split": split, "streaming": True}
    try:
        if config:
            ds = load_dataset(hf_id, config, **load_kw)
        else:
            ds = load_dataset(hf_id, **load_kw)
    except ConnectionError as e:
        raise RuntimeError(
            f"Cannot stream '{hf_id}' in offline mode. "
            f"Pre-save it from an internet-connected session:\n\n"
            f"  from datasets import load_dataset, Value, Dataset\n"
            f"  ds = load_dataset('{hf_id}', streaming=True, split='{split}')\n"
            f"  feats = ds.features.copy()\n"
            f"  for k, f in feats.items():\n"
            f"      if hasattr(f, 'names'): feats[k] = Value('string')\n"
            f"  Dataset.from_list(list(ds.cast(feats))).save_to_disk('/path/to/spatialqa')\n\n"
            f"Then run with: --dataset /path/to/spatialqa --split {split}"
        ) from e
    new_features = ds.features.copy()
    for key, feat in new_features.items():
        if hasattr(feat, "names"):  # ClassLabel
            new_features[key] = Value("string")
    ds = ds.cast(new_features)
    return _Dataset.from_list(list(ds))


def _load_dataset_resolved(dataset_arg: str, split_override, subset_override=None):
    """Load a HF dataset by slug, full HF id, or local path.

    split_override is None when the user did not pass --split, in which case the
    registry default (or 'test') is used.

    subset_override, when set, overrides the config/subset name (second positional
    arg to load_dataset) and takes precedence over the registry entry.

    Returns (dataset, hf_id_or_none) where hf_id_or_none is the HuggingFace repo
    id (e.g. "craigwu/vstar_bench") or None for local-path datasets.
    """
    from pathlib import Path as _Path
    p = _Path(dataset_arg)
    if p.exists() and p.is_dir():
        from datasets import load_from_disk
        loaded = load_from_disk(str(p))
        split = split_override or "test"
        return (loaded[split] if hasattr(loaded, "keys") else loaded), None
    slug = dataset_arg.split("/")[-1].lower().replace("-", "_")
    if slug in _DATASET_REGISTRY:
        entry = _DATASET_REGISTRY[slug]
        if entry is None:
            # Special-cased datasets with their own loaders.
            if slug == "visulogic":
                return _load_visulogic(), None
            if slug == "crossmath":
                return _load_crossmath(subset_override), "xuyige/CrossMath"
            if slug == "dailyclue":
                return _load_dailyclue(), "Crysun/DailyClue"
            if slug == "omnispatial":
                return _load_omnispatial(split_override), "qizekun/OmniSpatial"
            if slug == "virl39k":
                return _load_virl39k(), None
            if slug == "visual_cot":
                return _load_visual_cot(), None
            raise ValueError(f"No loader registered for dataset slug '{slug}'")
        hf_id, config, default_split = entry
        split = split_override or default_split
        config = subset_override or config
        if hf_id in _STREAMING_CAST_DATASETS:
            return _load_streaming_cast(hf_id, config, split), hf_id
        if config:
            return load_dataset(hf_id, config, split=split), hf_id
        return load_dataset(hf_id, split=split), hf_id
    split = split_override or "test"
    hf_id = dataset_arg if "/" in dataset_arg else None
    if hf_id in _STREAMING_CAST_DATASETS:
        return _load_streaming_cast(hf_id, subset_override, split), hf_id
    if subset_override:
        return load_dataset(dataset_arg, subset_override, split=split), hf_id
    return load_dataset(dataset_arg, split=split), hf_id


def _resolve_image(image_or_path, hf_id):
    """Return a PIL Image, downloading from HF hub if image_or_path is a string path."""
    from PIL import Image as _PIL
    if isinstance(image_or_path, _PIL.Image):
        return image_or_path
    if isinstance(image_or_path, str):
        import os
        # ViRL39K images live inside images.zip; extract the requested one on demand.
        if _VIRL39K_IMG_ROOT and image_or_path.startswith(_VIRL39K_IMG_ROOT):
            image_or_path = _virl39k_ensure_image(image_or_path)
        # Visual-CoT images live inside the combined image tar; extract on demand.
        if _VISCOT_IMG_ROOT and image_or_path.startswith(_VISCOT_IMG_ROOT):
            image_or_path = _viscot_ensure_image(image_or_path)
        if os.path.isabs(image_or_path) or os.path.exists(image_or_path):
            return _PIL.open(image_or_path)
        if hf_id is not None:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=hf_id, filename=image_or_path, repo_type="dataset")
            return _PIL.open(local)
        raise ValueError(f"Cannot resolve image path '{image_or_path}': not a local file and no HF repo id known.")
    return image_or_path  # passthrough for PIL images already decoded
SYSTEM_PROMPT = (
    "You are a meticulous and precise AI assistant, an expert in visual reasoning. "
    "Your primary goal is to solve the user's query by providing a detailed, step-by-step thought process.\n"
    "First, share your detailed reasoning inside <think>...</think> tags. "
    "Then, provide only your final answer inside <answer>...</answer> tags."
)
STEP_SYSTEM_PROMPT = (
    "You are a meticulous and precise AI assistant, an expert in visual reasoning. "
    "Your primary goal is to solve the user's query by providing a detailed, step-by-step thought process.\n"
    "First, share your detailed reasoning inside <think>...</think> tags. "
    "Within your reasoning, break it into explicit steps using <step>...</step> tags (e.g. <step>Observe the image...</step>, <step>Conclude that...</step>). "
    "Then, provide only your final answer inside <answer>...</answer> tags."
)
BB_SYSTEM_PROMPT = (
    "You are a meticulous and precise AI assistant, an expert in visual reasoning. "
    "Your primary goal is to solve the user's query by providing a detailed, step-by-step thought process.\n"
    "First, share your detailed reasoning inside <think>...</think> tags. "
    "The reasoning must be a numbered list of steps (1. ... 2. ... etc.). "
    "At the beginning of each step, output either <look> or <dont_look> to indicate whether that step will refer to a location in the image. "
    "A step that starts with <look> must refer to at least one specific image region and include a bounding box for it. "
    "A step that starts with <dont_look> must not reference any image location.\n"
    "Whenever a step refers to a location in the image, "
    "output a bounding box of that image region right before mentioning it. "
    "The bounding box format should be <box> [x1,y1,x2,y2] </box>, where [x1, y1] denotes the top-left corner and "
    "[x2, y2] the bottom-right corner of the bounding box, and all four are relative values (between 0 and 1). "
    'For example: "1. <look> I see the <box> [0.65,0.53,0.68,0.55] </box> door handle is on the right side.". '
    "Then, provide only your final answer inside <answer>...</answer> tags."
)
POD_SYSTEM_PROMPT = (
    "You are a meticulous and precise AI assistant, an expert in visual reasoning. "
    "Your primary goal is to solve the user's query through a detailed, numbered thought process "
    "grounded in image observations and logical deductions.\n"
    "Structure your reasoning as a numbered list of steps. "
    "Each step must be exactly one of the following types:\n"
    "- <plan> ... </plan> — A step in which you describe what you intend to examine or do next. "
    "Must not contain new image observations or deductions from prior steps. Always begins with \"I will\".\n"
    "- <observe> ... </observe> — A step in which you state something you see in the image. "
    "State the observation plainly, without interpretation or analysis. Always begins with \"I see\".\n"
    "- <deduce> ... </deduce> — A step in which you draw a logical conclusion from prior observation "
    "or deduction steps. Must not introduce new image observations. Always begins with "
    "\"Based on steps [LIST OF STEP NUMBERS], I deduce that...\", "
    "where the step numbers identify the steps this deduction is built on.\n"
    "After completing your reasoning, provide only your final answer inside <answer>...</answer> tags.\n\n"
    "Example:\n\n"
    "Question: Which way does this door open? "
    "A. The door opens outward, swinging to the left. "
    "B. The door opens inward, swinging to the left. "
    "C. The door opens inward, swinging to the right.\n\n"
    "Response:\n"
    "1. <plan> I will look at where the door meets the wall — if hinges are visible there, the door opens inward. </plan>\n"
    "2. <observe> I see that the area where the door connects to the wall is not visible in the image. </observe>\n"
    "3. <plan> I will look for other clues around the door. </plan>\n"
    "4. <observe> I see a toilet. </observe>\n"
    "5. <deduce> Based on steps [4], I deduce that this is a bathroom. </deduce>\n"
    "6. <deduce> Based on steps [5], I deduce that the door opens inward, since bathroom doors typically open inward. </deduce>\n"
    "7. <plan> I will look for the door handle, since its position indicates whether the door swings left or right. </plan>\n"
    "8. <observe> I see that the door handle is on the left side of the door. </observe>\n"
    "9. <deduce> Based on steps [6, 8], I deduce that the door opens inward and swings to the right — "
    "when a door opens inward and the handle is on the left, the door swings to the right. </deduce>\n"
    "10. <deduce> Based on steps [9], I deduce that the correct answer is C. </deduce>\n\n"
    "<answer> C </answer>"
)


_LOG_PREFIX = ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {_LOG_PREFIX}{msg}", flush=True)


def merge_rank_files(rank_paths: list, merged_path: Path) -> int:
    """Merge per-rank JSONL shards into one deduplicated file sorted by image_filename.

    Records already present in ``merged_path`` are folded in as well, so
    re-merging never destroys existing results.  This matters when an
    already-complete multi-GPU job is re-run: every sample is skipped on resume,
    so the rank shards are empty, and a naive overwrite would truncate the
    previously-complete merged file to nothing.  Rank-shard records take
    precedence over pre-existing merged records for the same sample_id.

    Returns the number of records written.
    """
    seen: dict = {}
    # Rank shards first so fresh results win over any stale merged copy.
    for p in rank_paths:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        sid = r.get("sample_id", r.get("id"))
                        if sid not in seen:
                            seen[sid] = r
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    # Fold in records already in the merged file that no shard re-produced.
    try:
        with open(merged_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    sid = r.get("sample_id", r.get("id"))
                    if sid not in seen:
                        seen[sid] = r
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    records = sorted(seen.values(), key=lambda r: r.get("image_filename", 0))
    with merged_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    return len(records)


def _aggregate_rank_progress(rank_paths: list) -> tuple[int, int]:
    """Count (n_done, n_correct) across all rank output files."""
    n_done = n_correct = 0
    for p in rank_paths:
        try:
            with open(p) as f:
                for line in f:
                    if line.strip():
                        n_done += 1
                        if json.loads(line).get("correct"):
                            n_correct += 1
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return n_done, n_correct


def _claim_next_sample(counter_path: Path) -> int:
    """Atomically claim and return the next sample index from a shared counter file."""
    import fcntl
    with open(counter_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            idx = int(f.read().strip())
            f.seek(0)
            f.write(str(idx + 1))
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return idx


def _increment_file_counter(path: Path) -> int:
    """Atomically increment a counter file and return the new value."""
    import fcntl
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            val = int(f.read().strip() or "0") + 1
            f.seek(0)
            f.write(str(val))
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return val


def _dynamic_sample_iter(ds, counter_path: Path):
    """Yield (global_index, row) pairs by atomically claiming indices from a shared counter."""
    n = len(ds)
    while True:
        i = _claim_next_sample(counter_path)
        if i >= n:
            break
        yield i, ds[i]


def parse_response(response: str) -> tuple[str, str]:
    """Return (reasoning, final_answer).

    Expected format (enforced by SYSTEM_PROMPT):
      <think>...</think> <answer>...</answer>

    Fallbacks for robustness:
      - If <answer> block is missing, take everything after </think>.
      - If <think> is present but </think> is absent, the response was truncated
        mid-generation; return ("", "") so the sample is marked incorrect.
      - If neither <think> nor <answer> is present, the response was truncated
        before any structured output; return ("", "") so the sample is marked
        incorrect (avoids false positives from single-letter ground truths
        appearing anywhere in a long truncated reasoning dump).
    """
    answer_m = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)

    # If the model opened <think> but never closed it, the response was truncated
    # mid-generation (e.g. hit max_new_tokens while still inside <think>).
    # Only bail out if there's also no <answer> block to fall back on.
    if re.search(r"<think>", response) and not re.search(r"</think>", response) and not answer_m:
        return "", ""

    think_m = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    reasoning = think_m.group(1).strip() if think_m else ""

    if answer_m:
        return reasoning, answer_m.group(1).strip()

    after_think = re.search(r"</think>(.*)", response, flags=re.DOTALL)
    if after_think:
        return reasoning, after_think.group(1).strip()

    # No <think> or <answer> tags at all — response was truncated before reaching
    # any structured output. Mark incorrect rather than using the raw dump as the
    # answer (which causes false positives when the ground-truth letter appears
    # incidentally in the reasoning text).
    return "", ""


# Option-letter ceiling for multiple-choice extraction. Covers up to 10 options
# (A-J), which is the widest choice set among our datasets (MMMU-Pro).
_MCQ_LETTERS = set(string.ascii_uppercase[:10])


def extract_mcq_choice(prediction: str, valid: set = _MCQ_LETTERS) -> str | None:
    """Extract the single option letter a model committed to, or None.

    A model *commits* to a letter only when it appears in a choice position:
      1. the answer *is* exactly that letter (optionally wrapped in ()/[]/** or
         with a trailing . ) :), possibly case-insensitive;
      2. right after an explicit cue — "answer is X", "option X", "option is X",
         "choose/select/pick X", "is X." / "is X," (letter then punctuation/EOL),
         or "\\boxed{X}";
      3. as a leading enumeration marker — "X.", "X:", "(X)", "X)" at the start.

    Everything else — refusals ("none of the options"), answers that discuss
    several options, or option *text* stated as prose (where a stray capital
    letter is not a choice) — returns None, so it is scored incorrect rather than
    matching whenever the GT letter happens to appear somewhere in the text.

    Cue/enumeration matches require an UPPERCASE option letter (letters[:], not
    [A-Za-z]): models write real choices in caps, so this filters out the English
    article "a" and mid-prose lowercase letters while still catching "answer is
    A". The exact-match case (1) stays case-insensitive since a lone "a" as the
    whole answer is a genuine choice.
    """
    if not prediction:
        return None
    p = prediction.strip()
    letters = "".join(sorted(valid))  # e.g. "ABCDEFGHIJ"
    # 1) Exact single letter, optionally wrapped in (), [], ** or trailing . ) :
    m = re.fullmatch(r"[\*\(\[]*([A-Za-z])[\*\)\]]*[.\):]?", p)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    low = p.lower()
    # 2) Explicit refusal — the model committed to no option.
    if "none of the" in low or "none of these" in low:
        return None
    # 3) Explicit choice cues, requiring an uppercase option letter. Cue words are
    #    matched case-insensitively (they may start a sentence, e.g. "Option I").
    for pat in (rf"[Aa]nswer\s*(?:is|:|=)\s*[\*\(\[]*([{letters}])\b",
                rf"\b[Oo]ption\s+(?:is\s+)?[\*\(\[]*([{letters}])\b",
                rf"\b(?:[Cc]hoose|[Ss]elect|[Pp]ick)\s+(?:option\s+)?[\*\(\[]*([{letters}])\b",
                rf"\b[Ii]s\s+(?:option\s+)?[\*\(\[]*([{letters}])(?=[.,\):]|\s*$)",
                rf"\\boxed\{{\s*([{letters}])\s*\}}"):
        m = re.search(pat, p)
        if m:
            return m.group(1).upper()
    # 4) Leading enumeration marker: "A.", "A:", "(A)", "A)".
    m = re.match(rf"[\*\(\[]*([{letters}])(?:[\*\)\]]+|[.:\)])", p)
    if m:
        return m.group(1).upper()
    return None


def parse_mcq_options(question: str) -> dict:
    """Map option letter -> normalized option text from a question's choice list.

    Recognizes enumerated option lines like "A. Yes", "(B) No", "C) Foo",
    "D: Bar". Text is lower-cased and stripped of a trailing period so it can be
    compared against a ground truth stated as option text.  Returns {} when the
    question has no such list (open-ended) or is None.
    """
    if not question:
        return {}
    opts = {}
    for m in re.finditer(r"^\s*[\(\[]?([A-Z])[\)\].:]\s+(.+?)\s*$", question, re.M):
        opts[m.group(1).upper()] = m.group(2).strip().lower().rstrip(".")
    return opts


def _resolve_mcq_letter(text: str, options: dict) -> str | None:
    """Resolve a raw answer/ground-truth string to an option letter, or None.

    First honours an explicit letter commitment (extract_mcq_choice); failing
    that, matches the answer *text* exactly against an option's text so that a
    model answering "No" resolves to whichever letter carries "No" in *this*
    sample (option order varies per sample).
    """
    letter = extract_mcq_choice(text, valid=set(options) or _MCQ_LETTERS)
    if letter is not None:
        return letter
    t = text.strip().lower().rstrip(".")
    matches = [ltr for ltr, otext in options.items() if otext == t]
    return matches[0] if len(matches) == 1 else None


def is_correct(prediction: str, ground_truth: str, question: str | None = None) -> bool:
    if not prediction or not ground_truth:
        return False
    g = ground_truth.strip()
    options = parse_mcq_options(question)

    # Reconcile letter answers against text ground truths (and vice-versa) using
    # the question's own option list, whose letter->text order varies per sample.
    # A ground truth is multiple-choice when it is a lone letter, or when it is
    # the text of one of the parsed options.
    gt_letter = None
    if len(g) == 1 and g.isalpha() and (not options or g.upper() in options):
        gt_letter = g.upper()
    elif options:
        gt_letter = _resolve_mcq_letter(g, options)
    if gt_letter is not None:
        # Compare the option the model committed to, not a raw substring (which
        # false-positives whenever the GT letter/word appears anywhere in a
        # verbose answer, e.g. "None of the options A, B, C, or D..." vs GT "D").
        return _resolve_mcq_letter(prediction, options) == gt_letter

    p = prediction.lower().strip()
    g = g.lower()
    return g in p or p in g


_WARNED_MISSING_FIELDS: set = set()


def get_field(row, names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    key = tuple(names)
    if key not in _WARNED_MISSING_FIELDS:
        _WARNED_MISSING_FIELDS.add(key)
        log(f"WARNING: none of {names} found in row (available keys: {list(row.keys())}); "
            f"defaulting to {default!r}")
    return default


QUESTION_FIELD_CANDIDATES = ["question", "query", "prompt", "problem"]
GROUND_TRUTH_FIELD_CANDIDATES = ["answer", "gt", "label", "ground_truth", "solution"]


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_preprocess_image(image, image_size=448, max_num=12):
    """Dynamic-tiling image preprocessing for InternVL3_5-style models that lack an HF processor."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    ratios = sorted(
        {(i, j) for n in range(1, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= max_num},
        key=lambda r: r[0] * r[1],
    )
    best = min(ratios, key=lambda r: abs(aspect - r[0] / r[1]))
    tw, th = image_size * best[0], image_size * best[1]
    resized = image.resize((tw, th))
    cols = best[0]
    tiles = [
        resized.crop(((k % cols) * image_size, (k // cols) * image_size,
                      ((k % cols) + 1) * image_size, ((k // cols) + 1) * image_size))
        for k in range(best[0] * best[1])
    ]
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return torch.stack([transform(t) for t in tiles])


def load_model(model_id: str, force_eager: bool = False, attn_impl: str | None = None,
               device_map=None):
    log(f"Loading model: {model_id}")

    # Detect PEFT/LoRA adapters: the directory has adapter_config.json but no config.json.
    # Load the base model first, then wrap it with the adapter.
    import os as _os, json as _json
    _adapter_cfg_path = _os.path.join(model_id, "adapter_config.json")
    if _os.path.isfile(_adapter_cfg_path):
        with open(_adapter_cfg_path) as _f:
            _adapter_cfg = _json.load(_f)
        _base_model_id = _adapter_cfg.get("base_model_name_or_path")
        log(f"  detected PEFT adapter; base model: {_base_model_id}")
        _base = load_model(_base_model_id, force_eager=force_eager, attn_impl=attn_impl,
                           device_map=device_map)
        from peft import PeftModel as _PeftModel
        log(f"  applying LoRA adapter from {model_id}")
        return _PeftModel.from_pretrained(_base, model_id)

    log("  efficiency: bfloat16 + device_map=auto + flash_attention_2")
    base = dict(device_map=device_map if device_map is not None else "auto",
                trust_remote_code=True)

    # mllama (Llama-3.2-Vision) loads with flash_attention_2 without error but
    # crashes at runtime because MllamaVisionAttention lacks is_causal.  Skip it.
    try:
        from transformers import AutoConfig as _AutoConfig
        _cfg = _AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        _model_type = getattr(_cfg, "model_type", "")
    except Exception:
        _model_type = ""
    if attn_impl is not None:
        _attn_impls = (attn_impl,)
    elif force_eager:
        _attn_impls = ("eager",)
    elif _model_type == "mllama":
        _attn_impls = ("sdpa", "eager", None)
    else:
        _attn_impls = ("flash_attention_2", "sdpa", "eager", None)

    def _from_pretrained(kwargs):
        errors = {}
        for cls in (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel):
            if cls is None:
                continue
            try:
                return cls.from_pretrained(model_id, **kwargs)
            except Exception as e:
                errors[cls.__name__] = f"{type(e).__name__}: {e}"
        details = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(f"No Auto class could load {model_id}. Errors: [{details}]")

    def _load(attn_impl, dtype_key, extra=None):
        kw = {**base, dtype_key: torch.bfloat16, **(extra or {})}
        if attn_impl is not None:
            kw["attn_implementation"] = attn_impl
        return _from_pretrained(kw)

    model = None
    for attn_impl in _attn_impls:
        for dtype_key in ("torch_dtype", "dtype"):
            try:
                model = _load(attn_impl, dtype_key)
                log(f"  attn_implementation={attn_impl or '<default>'}, dtype_kwarg={dtype_key}")
                break
            except (ImportError, ValueError, RuntimeError) as e:
                log(f"  {attn_impl or '<default>'}/{dtype_key} failed "
                    f"({type(e).__name__}: {e}); trying next")
        if model is not None:
            break

    if model is None:
        # Some models (e.g. InternVL3-Instruct) hardcode flash_attention_2 in
        # __init__ via a use_flash_attn=True default, overriding attn_implementation.
        # Passing use_flash_attn=False as a model kwarg forces eager attention.
        log("  retrying with use_flash_attn=False")
        for attn_impl in ("eager", None):
            for dtype_key in ("torch_dtype", "dtype"):
                try:
                    model = _load(attn_impl, dtype_key, extra={"use_flash_attn": False})
                    log(f"  loaded with use_flash_attn=False, "
                        f"attn_implementation={attn_impl or '<default>'}, dtype_kwarg={dtype_key}")
                    break
                except (ImportError, TypeError, ValueError, RuntimeError) as e:
                    log(f"  use_flash_attn=False/{attn_impl or '<default>'}/{dtype_key} failed "
                        f"({type(e).__name__}: {e})")
            if model is not None:
                break

    if model is None:
        # DeepSeek-VL2 and similar models have no auto_map in their config and
        # require a vendor-specific class from the deepseek_vl2 package.
        log("  retrying with DeepseekVLV2ForCausalLM")
        try:
            from deepseek_vl2.models import DeepseekVLV2ForCausalLM as _DVLV2
            model = _DVLV2.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            log("  loaded via DeepseekVLV2ForCausalLM")
        except Exception as e:
            log(f"  DeepseekVLV2ForCausalLM failed ({type(e).__name__}: {e})")

    if model is None:
        raise RuntimeError("No supported loading configuration found for this model.")
    model.eval()
    return model


def _encode_image_data_uri(image) -> str:
    """PIL image → a `data:` URI with a base64-encoded PNG payload, as expected by
    the OpenAI `image_url` content field."""
    import base64
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def call_chat_api(api_base, api_key, model, system_prompt_text, question, image,
                  max_tokens=8192, temperature=None, top_p=None,
                  reasoning_budget=None, enable_thinking=False,
                  timeout=1800, max_retries=5):
    """Send one image+text turn to an OpenAI-compatible /chat/completions endpoint
    (e.g. a locally served NVIDIA NIM or the hosted NVIDIA API at
    https://integrate.api.nvidia.com/v1) and return (response_text, completion_tokens).

    Uses only the stdlib (urllib) so no extra dependency is needed. The returned
    text is normalised to the <think>...</think><answer>...</answer> shape that
    parse_response() expects: if the server emits chain-of-thought in a separate
    `reasoning_content` field (as vLLM reasoning parsers do) and the visible content
    has no <think> block, the reasoning is wrapped back in so parsing is unchanged.

    top_p, reasoning_budget, and enable_thinking mirror the params in the hosted
    NVIDIA API example for nvidia/nemotron-3-nano-omni-30b-a3b-reasoning. In the
    OpenAI SDK these last two are passed via extra_body; extra_body simply merges
    its keys into the top-level request JSON, so here they are set on the payload
    directly (chat_template_kwargs={"enable_thinking": ...} and reasoning_budget).
    """
    import urllib.error
    import urllib.request

    messages = []
    if system_prompt_text is not None:
        messages.append({"role": "system", "content": system_prompt_text})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": _encode_image_data_uri(image)}},
            {"type": "text", "text": question},
        ],
    })
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if reasoning_budget is not None:
        payload["reasoning_budget"] = reasoning_budget

    url = api_base.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0].get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if reasoning and "<think>" not in content:
                content = f"<think>{reasoning}</think>\n{content}"
            completion_tokens = (body.get("usage") or {}).get("completion_tokens", 0)
            return content, completion_tokens
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            last_err = e
            wait = 2 ** attempt
            log(f"  [api] request failed ({type(e).__name__}: {e}); "
                f"retry {attempt + 1}/{max_retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API request to {url} failed after {max_retries} attempts: {last_err}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=None,
                   help="Run inference against an OpenAI-compatible HTTP endpoint (e.g. a locally "
                        "served NVIDIA NIM at http://<host>:8000/v1, or the hosted NVIDIA API at "
                        "https://integrate.api.nvidia.com/v1) instead of loading a local model. "
                        "When set, --model is the served model id (see GET /v1/models). "
                        "Incompatible with --saliency / --enforce-thinking-budget.")
    p.add_argument("--api-key", default=None,
                   help="Bearer token for --api-base. Defaults to the NVIDIA_API_KEY env var if set, "
                        "else 'EMPTY' (local NIMs ignore it; the hosted NVIDIA API requires a real key).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature for --api-base requests (default: the server's own).")
    p.add_argument("--top-p", type=float, default=None,
                   help="Nucleus sampling top_p for --api-base requests (default: the server's own).")
    p.add_argument("--reasoning-budget", type=int, default=None,
                   help="Reasoning token budget for --api-base requests (NVIDIA NIM extra_body field). "
                        "Only sent when set, e.g. --reasoning-budget 16384.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Enable the served model's thinking/reasoning mode for --api-base requests "
                        "(sends chat_template_kwargs={'enable_thinking': true}).")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=None,
                   help="Dataset split override. Defaults to the registry default for known datasets, 'test' otherwise.")
    p.add_argument("--subset", default=None,
                   help="Dataset subset/config name (second positional arg to load_dataset). "
                        "Overrides the registry entry, e.g. --subset vision for MMMU/MMMU_Pro.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run on only the first N samples (default: full split).")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Generation budget; high by design to fit long reasoning chains.")
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system-prompt", default=None,
                   help="Override the system prompt text. Defaults to the built-in SYSTEM_PROMPT.")
    p.add_argument("--no-system-prompt", action="store_true",
                   help="Omit the system message entirely.")
    p.add_argument("--bb-system-prompt", action="store_true",
                   help="Use BB_SYSTEM_PROMPT, which asks the model to ground mentions with bounding boxes.")
    p.add_argument("--step-system-prompt", action="store_true",
                   help="Use STEP_SYSTEM_PROMPT, which asks the model to wrap each reasoning step in <step> tags.")
    p.add_argument("--pod-system-prompt", action="store_true",
                   help="Use POD_SYSTEM_PROMPT, which asks the model to structure reasoning as numbered "
                        "<plan>, <observe>, and <deduce> steps.")
    p.add_argument("--no-strip-direct-answer-suffix", dest="strip_direct_answer_suffix",
                   action="store_false", default=True,
                   help="Disable stripping of direct-answer instructions embedded in RealWorldQA questions "
                        "(e.g. 'Please answer directly with only the letter…'). Stripping is on by default "
                        "so that system-prompt reasoning instructions are not overridden.")
    p.add_argument("--merge", action="store_true",
                   help="Instead of running inference, merge existing .rankN.jsonl shards for --output "
                        "into a single sorted file and exit.")
    p.add_argument("--rank", type=int, default=0,
                   help="Rank of this process (0-indexed). Use with --world-size to shard the dataset "
                        "across multiple processes / GPUs (default: 0).")
    p.add_argument("--world-size", type=int, default=1,
                   help="Total number of parallel processes. Each process handles "
                        "dataset[rank::world_size] samples (default: 1, i.e. no sharding).")
    p.add_argument("--gpu", type=int, default=None,
                   help="Pin this process to a specific GPU index by setting CUDA_VISIBLE_DEVICES "
                        "before model loading. Useful when running one process per GPU.")
    p.add_argument("--debug", action="store_true",
                   help="Print extra diagnostics per sample: raw token IDs, decoded prompt/full-output, "
                        "and decoding-path taken. Always active when response is empty.")
    p.add_argument("--saliency", action="store_true",
                   help="Extract and save per-sample saliency maps (requires attn_implementation=eager; "
                        "slower than default). Maps are saved as .npy files under "
                        "{output_stem}_saliency/. Only supported for Qwen-family models.")
    p.add_argument("--no-value-weighting", dest="value_weighting", action="store_false", default=True,
                   help="Do not multiply attention weights by value-state L2 norms when computing "
                        "saliency maps (default: weighting is on).")
    p.add_argument("--head-reduction", default="mean", choices=["mean", "min", "max"],
                   help="How to reduce across attention heads when computing saliency (default: mean).")
    p.add_argument("--layer-reduction", default="sum", choices=["sum", "min", "max"],
                   help="How to reduce across transformer layers when computing saliency (default: sum).")
    p.add_argument("--enforce-thinking-budget", action="store_true",
                   help="Two-stage generation: generate up to the thinking budget, force-close "
                        "</think> if the model did not stop on its own, then generate the answer "
                        "in a second pass. Guarantees an <answer> block at the cost of a second "
                        "generate() call per sample. Off by default.")
    p.add_argument("--force-answer", action="store_true",
                   help="If the response has no closed <answer>...</answer> block (e.g. the model "
                        "hit max_new_tokens or fell into a repetition loop), re-prompt with the "
                        "original context + the generated response + a forced '<answer>' opener, "
                        "and generate the answer in a second pass. The two generations are glued "
                        "into a single response so downstream parsing sees one <answer> block. "
                        "Off by default.")
    p.add_argument("--force-answer-max-new-tokens", type=int, default=512,
                   help="Token budget for the second (answer-forcing) pass when --force-answer "
                        "triggers (default: 512).")

    # --- VGA (Vision-Guided Attention, arXiv:2511.20032) ----------------------
    # Training-free: stock weights plus an inference-time patch.  See
    # wiki/vga-implementation.md and vlm/vga.py.
    g = p.add_argument_group("VGA (vision-guided attention)")
    g.add_argument("--vga", action="store_true",
                   help="Steer answer tokens' attention toward the visual patches that carry "
                        "the question's objects. Training-free; batch size 1 only.")
    g.add_argument("--vga-beta", type=float, default=0.2,
                   help="Guidance strength (default: 0.2; the paper uses 0.25 for POPE and "
                        "for larger models). 0 is exactly the identity.")
    g.add_argument("--vga-start-layer", type=int, default=4,
                   help="First decoder layer to inject into (default: 4, Qwen2.5-VL's value). "
                        "MUST be re-tuned for Qwen3-VL — see analysis/vga_layer_scan.py.")
    g.add_argument("--vga-end-layer", type=int, default=16,
                   help="Layer to stop at, exclusive (default: 16).")
    g.add_argument("--vga-end-inclusive", action="store_true",
                   help="Treat --vga-end-layer as inclusive instead.")
    g.add_argument("--vga-mode", default="auto", choices=["auto", "object", "agnostic"],
                   help="'object' = VSC over the question's objects; 'agnostic' = VSS entropy "
                        "over the top-K vocabulary; 'auto' (default) uses VSC and falls back "
                        "to VSS when no object can be extracted.")
    g.add_argument("--vga-topk", type=int, default=10,
                   help="K for the VSS entropy (default: 10).")
    g.add_argument("--vga-vss-invert", action="store_true",
                   help="Use 1 - normalised entropy for VSS. The paper's prose and upstream's "
                        "code disagree on this sign; see salience_map() in vlm/vga.py.")
    g.add_argument("--vga-head-balancing", default="simg", choices=["simg", "none"],
                   help="Down-weight guidance on heads already doing visual work (default: simg).")
    g.add_argument("--vga-attn-norm", action="store_true",
                   help="Convex-blend variant instead of the additive update. Upstream defaults "
                        "it off; leave it off.")
    g.add_argument("--vga-no-sparsity-scaling", dest="vga_sparsity_scaling",
                   action="store_false", default=True,
                   help="Do not scale beta by G's L0 fraction.")
    g.add_argument("--vga-object-variants", default="both", choices=["both", "plain"],
                   help="'plain' scores only the object's literal first token; 'both' (default) "
                        "also tries the space-prefixed and capitalised forms and keeps the max.")
    g.add_argument("--vga-pvg", action="store_true",
                   help="Progressive Visual Guidance: after each generated token, suppress the "
                        "regions it already described. For captioning; pointless for short-answer VQA.")
    g.add_argument("--vga-lambda", type=float, default=0.02,
                   help="PVG suppression rate (default: 0.02).")
    g.add_argument("--vga-no-prefill-reuse", dest="vga_prefill_reuse",
                   action="store_false", default=True,
                   help="Re-encode the prompt for generation instead of resuming from the "
                        "guidance pass's KV cache. Costs a second prefill per sample and starts "
                        "injecting one token later; use it if the resume path misbehaves.")

    args = p.parse_args()

    if args.merge:
        out = Path(args.output)
        rank_paths = sorted(out.parent.glob(f"{out.stem}.rank*{out.suffix}"))
        if not rank_paths:
            print(f"No rank shard files found matching {out.parent}/{out.stem}.rank*{out.suffix}")
            return
        n = merge_rank_files(rank_paths, out)
        n_correct = sum(1 for r in [json.loads(l) for p in rank_paths
                                    for l in p.open() if l.strip()]
                        if r.get("correct"))
        with out.open() as f:
            all_r = [json.loads(l) for l in f if l.strip()]
        n_correct = sum(1 for r in all_r if r.get("correct"))
        print(f"Merged {len(rank_paths)} shard(s) → {n} records → {out}")
        if n:
            print(f"Accuracy: {n_correct}/{n} = {n_correct/n:.4f}")
        for p in rank_paths:
            p.unlink()
            print(f"Deleted rank shard: {p}")
        return

    if args.gpu is not None:
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        log(f"Pinned to GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")

    if args.world_size > 1:
        global _LOG_PREFIX
        _LOG_PREFIX = f"[r{args.rank}] "
        out = Path(args.output)
        args.output = str(out.parent / f"{out.stem}.rank{args.rank}{out.suffix}")
        log(f"Dynamic queue mode: rank={args.rank}/{args.world_size}, output={args.output}")

    if args.no_system_prompt:
        system_prompt_text = None
        prompt_type = "none"
    elif args.bb_system_prompt:
        system_prompt_text = BB_SYSTEM_PROMPT
        prompt_type = "bb"
    elif args.step_system_prompt:
        system_prompt_text = STEP_SYSTEM_PROMPT
        prompt_type = "step"
    elif args.pod_system_prompt:
        system_prompt_text = POD_SYSTEM_PROMPT
        prompt_type = "pod"
    elif args.system_prompt is not None:
        system_prompt_text = args.system_prompt
        prompt_type = "custom"
    else:
        system_prompt_text = SYSTEM_PROMPT
        prompt_type = "default"

    log(f"Prompt type: {prompt_type}")

    # API mode: talk to an OpenAI-compatible endpoint (e.g. a locally served NIM)
    # instead of loading a local model.  All model/processor setup is skipped; the
    # per-sample generation branch calls the HTTP API to produce `response`, and
    # everything downstream (parsing, scoring, output, resume, sharding) is shared.
    api_mode = args.api_base is not None
    if api_mode:
        if args.saliency or args.enforce_thinking_budget:
            p.error("--api-base is incompatible with --saliency and --enforce-thinking-budget "
                    "(both require local token/attention access).")
        if args.force_answer:
            log("WARNING: --force-answer is not supported in API mode; ignoring it.")
        if args.api_key is None:
            import os as _os
            args.api_key = _os.environ.get("NVIDIA_API_KEY", "EMPTY")
        log(f"API mode: endpoint={args.api_base} model={args.model!r} "
            f"temperature={args.temperature} top_p={args.top_p} "
            f"enable_thinking={args.enable_thinking} reasoning_budget={args.reasoning_budget}")
        model = processor = tokenizer = None
        deepseek_vl2_mode = internvl_legacy = False
        system_content_as_list = system_prompt_inline = False
        stop_token_ids = []
    else:
        set_seed(args.seed)

        model = load_model(args.model, force_eager=args.saliency)

        # Option 1: saliency hooks only need the LM in eager mode.  Switch the vision
        # encoder back to SDPA so its full attention matrix is never materialised.
        if args.saliency and hasattr(model, "visual") and hasattr(model.visual, "config"):
            vis_cfg = model.visual.config
            if getattr(vis_cfg, "_attn_implementation", None) == "eager":
                vis_cfg._attn_implementation = "sdpa"
                log("  Patched vision encoder: eager → sdpa (LM stays eager for saliency hooks)")

        log("Loading processor")
        # DeepSeek-VL2 has no auto_map so AutoProcessor fails; use vendor class directly.
        try:
            from deepseek_vl2.models import DeepseekVLV2ForCausalLM as _DVLV2
            _is_deepseek_vl2 = isinstance(model, _DVLV2)
        except ImportError:
            _is_deepseek_vl2 = False
        if _is_deepseek_vl2:
            from deepseek_vl2.models import DeepseekVLV2Processor as _DVLV2Proc
            processor = _DVLV2Proc.from_pretrained(args.model)
            log("  loaded DeepseekVLV2Processor")
        else:
            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        # InternVL3 (loaded via AutoModel with trust_remote_code) requires this token
        # ID to be set manually; other models don't have the attribute.
        if hasattr(model, 'img_context_token_id') and model.img_context_token_id is None:
            tokenizer = getattr(processor, 'tokenizer', processor)
            img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            model.img_context_token_id = img_ctx_id
            log(f"  set img_context_token_id={img_ctx_id}")

        # DeepSeek-VL2: uses its own processor API (conversations + system_prompt kwarg).
        deepseek_vl2_mode = _is_deepseek_vl2
        if deepseek_vl2_mode:
            deepseek_vl2_tokenizer = processor.tokenizer
            # model.language is loaded as a sub-module so generation_config is None;
            # newer transformers' generate() requires it to exist.
            if model.language.generation_config is None:
                from transformers import GenerationConfig
                model.language.generation_config = GenerationConfig()
            log("  DeepSeek-VL2 mode")

        # InternVL3_5-style models ship without a HuggingFace vision processor
        # (no preprocessor_config.json). They must be driven through model.chat()
        # with manually pre-processed pixel_values.
        internvl_legacy = (not deepseek_vl2_mode
                           and hasattr(model, 'chat')
                           and not hasattr(processor, 'image_processor'))
        if internvl_legacy:
            internvl_tokenizer = getattr(processor, 'tokenizer', processor)
            internvl_image_size = int(getattr(model.config, 'force_image_size', None) or 448)
            internvl_max_patches = int(getattr(model.config, 'max_dynamic_patch', None) or 12)
            log(f"  InternVL-legacy mode: image_size={internvl_image_size} max_patches={internvl_max_patches}")

        # Detect whether this processor's chat template expects system content as a
        # list-of-dicts (e.g. Qwen) or a plain string (e.g. InternVL3).
        # Not applicable for deepseek_vl2_mode (uses a different conversation API).
        system_content_as_list = False
        if not deepseek_vl2_mode and not internvl_legacy:
            _probe_text = system_prompt_text or "test"
            _probe_list = [
                {"role": "system", "content": [{"type": "text", "text": _probe_text}]},
                {"role": "user", "content": [{"type": "text", "text": "test"}]},
            ]
            try:
                processor.apply_chat_template(_probe_list, tokenize=False, add_generation_prompt=False)
                system_content_as_list = True
            except Exception:
                system_content_as_list = False

        # Some processors (e.g. Molmo2) don't support a system role at all.
        # Their apply_chat_template(tokenize=True) iterates over every message's
        # content as a list-of-dicts to extract visuals, and crashes on a plain
        # string.  Detect this once with a dummy image so we can inline the system
        # prompt into the user text instead.
        system_prompt_inline = False
        if (not system_content_as_list and not deepseek_vl2_mode and not internvl_legacy
                and system_prompt_text is not None):
            try:
                from PIL import Image as _PIL_Image
                _dummy = _PIL_Image.new("RGB", (8, 8))
                _probe2 = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": [
                        {"type": "image", "image": _dummy},
                        {"type": "text", "text": "test"},
                    ]},
                ]
                processor.apply_chat_template(
                    _probe2, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
            except Exception:
                system_prompt_inline = True

        log(f"  system message content format: "
            f"{'list' if system_content_as_list else 'inline' if system_prompt_inline else 'string'}")
        log(f"  system prompt: {'(none)' if system_prompt_text is None else repr(system_prompt_text[:80] + ('...' if len(system_prompt_text) > 80 else ''))}")

        # Build a list of stop token IDs covering both the standard EOS and any
        # chat-template turn-ending tokens (e.g. <|im_end|> for InternVL3/Qwen).
        tokenizer = getattr(processor, 'tokenizer', processor)
        stop_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
        for tok in ('<|im_end|>', '<|endoftext|>', '</s>'):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)
        stop_token_ids = sorted(stop_token_ids)
        log(f"  stop_token_ids={stop_token_ids}")

    log(f"Loading dataset: {args.dataset} [split={args.split or 'registry default'}]"
        + (f" [subset={args.subset}]" if args.subset else ""))
    ds, _dataset_hf_id = _load_dataset_resolved(args.dataset, args.split, args.subset)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    n_total_global = len(ds)
    _all_rank_outputs = None
    _base_stem = None
    _counter_path = None
    _done_counter_path = None
    _counter_ready_path = None
    if args.world_size > 1:
        out_p = Path(args.output)
        # args.output already has .rankN appended; strip it to get the base stem
        _base_stem = re.sub(r"\.rank\d+$", "", out_p.stem)
        _all_rank_outputs = [
            out_p.parent / f"{_base_stem}.rank{r}{out_p.suffix}"
            for r in range(args.world_size)
        ]
        _counter_path = out_p.parent / f"{_base_stem}.counter"
        _done_counter_path = out_p.parent / f"{_base_stem}.done_ranks"
        _counter_ready_path = out_p.parent / f"{_base_stem}.counter.ready"
        if args.rank == 0:
            with open(_counter_path, 'w') as _f:
                _f.write("0")
            with open(_done_counter_path, 'w') as _f:
                _f.write("0")
            _counter_ready_path.touch()
            log(f"  Initialized dynamic work queue over {n_total_global} samples")
        else:
            log(f"  Waiting for work queue initialization by rank 0...")
            _t0_wait = time.time()
            while not _counter_ready_path.exists():
                time.sleep(0.1)
                if time.time() - _t0_wait > 60:
                    raise RuntimeError(f"Timed out waiting for work queue: {_counter_ready_path}")
            log(f"  Work queue ready")
    log(f"  {n_total_global} samples to process")
    log(f"  columns: {ds.column_names}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    saliency_dir = None
    if args.saliency:
        from selfsal.models.introspect import prepare_decode_saliency as _prepare_decode_saliency
        from selfsal.models.introspect import finalize_decode_saliency as _finalize_decode_saliency
        saliency_dir = Path("results/saliency/per_token") / out_path.stem
        saliency_dir.mkdir(parents=True, exist_ok=True)
        log(f"Saliency maps will be saved to {saliency_dir}")

    # --- VGA: wrap model.generate so every sample is guided ---------------------
    vga = None
    if args.vga:
        if model is None:
            raise ValueError("--vga needs a local model; it cannot patch an API endpoint.")
        from baselines.vga.vga import VGAConfig, install as _install_vga
        vga = _install_vga(model, processor, VGAConfig(
            beta=args.vga_beta,
            start_layer=args.vga_start_layer,
            end_layer=args.vga_end_layer,
            end_layer_inclusive=args.vga_end_inclusive,
            mode=args.vga_mode,
            topk=args.vga_topk,
            vss_invert=args.vga_vss_invert,
            head_balancing=args.vga_head_balancing,
            attn_norm=args.vga_attn_norm,
            sparsity_scaling=args.vga_sparsity_scaling,
            object_variants=args.vga_object_variants,
            pvg=args.vga_pvg,
            lam=args.vga_lambda,
            prefill_reuse=args.vga_prefill_reuse,
        ))

    # --- Cross-mode resume: migrate existing results to the expected output file ---
    #
    # Case 1: running with world_size=1 but rank shards exist from a prior multi-GPU
    # run.  Merge all shards into the flat output file so the normal resume logic
    # picks them up.
    if args.world_size == 1 and not out_path.exists():
        rank_shards = sorted(out_path.parent.glob(f"{out_path.stem}.rank*{out_path.suffix}"))
        if rank_shards:
            n_migrated = merge_rank_files(rank_shards, out_path)
            log(f"[resume] migrated {n_migrated} records from {len(rank_shards)} rank shard(s) "
                f"into {out_path} for single-GPU resume")

    # Case 2: running with world_size>1 but only the flat (merged) file exists from a
    # prior single-GPU run.  With the dynamic queue any rank may process any sample,
    # so we populate done_ids directly from the flat file rather than seeding per-rank
    # shards.  This is done after the per-shard done_ids load below.

    # Read existing outputs and build a 'done' set so resumed jobs skip completed items.
    done_ids: set = set()
    n_correct_so_far = 0
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_ids.add(rec["sample_id"])
                    n_correct_so_far += int(rec.get("correct", False))
        acc_str = f"{n_correct_so_far/len(done_ids):.3f}" if done_ids else "n/a"
        log(f"[resume] found {len(done_ids)} completed items "
            f"({n_correct_so_far} correct, acc={acc_str}) "
            f"in {out_path}; will skip them")

    # Case 2: in multi-GPU mode, also load done_ids from sibling rank shards so
    # that a resumed run doesn't re-process samples another rank already completed.
    if args.world_size > 1 and _all_rank_outputs is not None:
        _extra_siblings = 0
        for _sibling_path in _all_rank_outputs:
            if _sibling_path == out_path:
                continue  # already loaded above
            try:
                with open(_sibling_path) as _sf:
                    for _line in _sf:
                        _line = _line.strip()
                        if _line:
                            _rec = json.loads(_line)
                            if _rec["sample_id"] not in done_ids:
                                done_ids.add(_rec["sample_id"])
                                n_correct_so_far += int(_rec.get("correct", False))
                                _extra_siblings += 1
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        if _extra_siblings:
            log(f"[resume] also loaded {_extra_siblings} records from sibling rank shards")

    # Case 3 (continued): load done_ids from the flat merged file so that a
    # multi-GPU run can resume where a prior single-GPU run left off.
    if args.world_size > 1 and _base_stem is not None:
        _flat_path = out_path.parent / (_base_stem + out_path.suffix)
        if _flat_path.exists() and _flat_path != out_path:
            _extra = 0
            with _flat_path.open() as _ff:
                for _line in _ff:
                    _line = _line.strip()
                    if _line:
                        _rec = json.loads(_line)
                        if _rec["sample_id"] not in done_ids:
                            done_ids.add(_rec["sample_id"])
                            n_correct_so_far += int(_rec.get("correct", False))
                            _extra += 1
            if _extra:
                log(f"[resume] also loaded {_extra} records from {_flat_path.name} "
                    f"(prior single-GPU run)")

    log(f"Starting inference (max_new_tokens={args.max_new_tokens})")
    n_done_so_far = len(done_ids)
    t0 = time.time()

    with out_path.open("a") as out_f:
        _sample_iter = (_dynamic_sample_iter(ds, _counter_path)
                        if args.world_size > 1 else enumerate(ds))
        for i, row in _sample_iter:
            # decoded_image (PIL) takes priority over image (may be a filename string or list)
            _raw_image = get_field(row, ["decoded_image", "image", "img"])
            if isinstance(_raw_image, list):
                _raw_image = _raw_image[0]
            image = _resolve_image(_raw_image, _dataset_hf_id)
            if hasattr(image, "convert") and image.mode != "RGB":
                image = image.convert("RGB")
            question = get_field(row, QUESTION_FIELD_CANDIDATES, default="")
            gt = get_field(row, GROUND_TRUTH_FIELD_CANDIDATES)
            sample_id = get_field(row, ["id", "qid", "question_id", "pid"], default=i)

            # RealWorldQA embeds a direct-answer instruction that overrides any
            # step-by-step system prompt.  Strip it so the system prompt wins.
            if args.strip_direct_answer_suffix and "realworldqa" in args.dataset.lower():
                for _suffix in (
                    "\nPlease answer directly with only the letter of the correct option and nothing else.",
                    "\nPlease answer directly with a single word or number.",
                ):
                    if question.endswith(_suffix):
                        question = question[: -len(_suffix)]
                        break

            # Append answer choices when present (e.g. MathVista multi-choice questions)
            choices = get_field(row, ["choices", "options"])
            if choices:
                if isinstance(choices, str):
                    try:
                        choices = ast.literal_eval(choices)
                    except Exception:
                        choices = None
            if choices:
                labels = string.ascii_uppercase
                question = question + "\n" + "\n".join(
                    f"{labels[j]}. {c}" for j, c in enumerate(choices)
                )

            if sample_id in done_ids:
                continue

            sal_path = None  # set in standard HF path when --saliency is on

            if api_mode:
                # Produce `response` via the OpenAI-compatible HTTP endpoint; the
                # system prompt + question + image map straight onto a chat turn.
                t_preprocess = 0.0
                _t_gen = time.time()
                response, n_generated = call_chat_api(
                    args.api_base, args.api_key, args.model,
                    system_prompt_text, question, image,
                    max_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, reasoning_budget=args.reasoning_budget,
                    enable_thinking=args.enable_thinking,
                )
                t_generate = time.time() - _t_gen
                prompt_len = 0
                if args.debug or not response:
                    log(f"  [debug sample={sample_id}] api response_len={len(response)} "
                        f"completion_tokens={n_generated}")
            elif deepseek_vl2_mode:
                conversations = [
                    {
                        "role": "<|User|>",
                        "content": f"<image>\n{question}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                t_preprocess = time.time()
                prepare_inputs = processor(
                    conversations=conversations,
                    images=[image],
                    force_batchify=True,
                    system_prompt=system_prompt_text or "",
                ).to(model.device)
                t_preprocess = time.time() - t_preprocess

                t_generate = time.time()
                with torch.inference_mode():
                    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                    output_ids = model.language.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=deepseek_vl2_tokenizer.eos_token_id,
                        bos_token_id=deepseek_vl2_tokenizer.bos_token_id,
                        eos_token_id=deepseek_vl2_tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                t_generate = time.time() - t_generate
                response = deepseek_vl2_tokenizer.decode(
                    output_ids[0].cpu().tolist(), skip_special_tokens=True
                )
                prompt_len = prepare_inputs.input_ids.shape[1]
                n_generated = output_ids.shape[1]

                if args.debug or not response:
                    log(f"  [debug sample={sample_id}] deepseek-vl2 response_len={len(response)}")
            elif internvl_legacy:
                t_preprocess = time.time()
                pixel_values = _internvl_preprocess_image(
                    image, internvl_image_size, internvl_max_patches
                ).to(model.device, dtype=torch.bfloat16)
                t_preprocess = time.time() - t_preprocess

                orig_sys = model.system_message
                if system_prompt_text is not None:
                    model.system_message = system_prompt_text
                gen_cfg = {'max_new_tokens': args.max_new_tokens}
                t_generate = time.time()
                response = model.chat(internvl_tokenizer, pixel_values, question, gen_cfg)
                t_generate = time.time() - t_generate
                model.system_message = orig_sys
                prompt_len = 0
                n_generated = 0

                if args.debug or not response:
                    log(f"  [debug sample={sample_id}] internvl-legacy response_len={len(response)}")
            else:
                messages = []
                if system_prompt_text is not None and not system_prompt_inline:
                    system_content = ([{"type": "text", "text": system_prompt_text}]
                                      if system_content_as_list else system_prompt_text)
                    messages.append({"role": "system", "content": system_content})
                user_text = (
                    f"{system_prompt_text}\n\n{question}"
                    if system_prompt_inline and system_prompt_text else question
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                })

                t_preprocess = time.time()
                thinking_budget = max(0, args.max_new_tokens - 512)
                _apply_kw = dict(
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if args.enforce_thinking_budget:
                    try:
                        inputs = processor.apply_chat_template(
                            messages, **_apply_kw, thinking_budget=thinking_budget,
                        ).to(model.device)
                    except TypeError:
                        # Processor does not support thinking_budget (non-Qwen3 models)
                        inputs = processor.apply_chat_template(
                            messages, **_apply_kw,
                        ).to(model.device)
                else:
                    inputs = processor.apply_chat_template(
                        messages, **_apply_kw,
                    ).to(model.device)
                t_preprocess = time.time() - t_preprocess

                prompt_len = inputs["input_ids"].shape[1]

                # VGA extracts objects from the question. We have the question
                # verbatim here, so hand it over rather than letting vga recover
                # it by decoding the prompt back out of the token ids.
                if vga is not None:
                    vga.set_question(question)

                # Option 2: register decode-time saliency hooks before generate() so
                # we collect attention during decoding instead of a separate fwd pass.
                _sal_state = None
                if saliency_dir is not None:
                    if args.enforce_thinking_budget:
                        log(f"  [saliency] sample={sample_id}: skipped (incompatible with --enforce-thinking-budget)")
                    else:
                        try:
                            _sal_state = _prepare_decode_saliency(model, processor, inputs)
                        except Exception as _sal_err:
                            log(f"  [saliency] sample={sample_id} prepare failed: {_sal_err}")

                t_generate = time.time()
                if args.enforce_thinking_budget and thinking_budget > 0:
                    # Stage 1: thinking phase — generate up to thinking_budget tokens.
                    with torch.inference_mode():
                        think_ids = model.generate(
                            **inputs,
                            max_new_tokens=thinking_budget,
                            eos_token_id=stop_token_ids,
                        )
                    # Force-close thinking if the model did not stop on its own.
                    decoded_thinking = processor.batch_decode(
                        think_ids[:, prompt_len:],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    if "</think>" not in decoded_thinking:
                        close_ids = tokenizer.encode("</think>", add_special_tokens=False)
                        think_ids = torch.cat(
                            [think_ids, torch.tensor([close_ids], device=think_ids.device)], dim=1
                        )
                    # Stage 2: answer phase — generate up to the remaining token budget.
                    answer_budget = args.max_new_tokens - thinking_budget
                    stage2_inputs = {
                        **inputs,
                        "input_ids": think_ids,
                        "attention_mask": torch.ones(
                            think_ids.shape, dtype=inputs["attention_mask"].dtype,
                            device=think_ids.device,
                        ),
                    }
                    with torch.inference_mode():
                        output_ids = model.generate(
                            **stage2_inputs,
                            max_new_tokens=answer_budget,
                            eos_token_id=stop_token_ids,
                        )
                    _past_kv = None
                else:
                    with torch.inference_mode():
                        _gen_out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=stop_token_ids,
                            return_dict_in_generate=(_sal_state is not None),
                        )
                    if _sal_state is not None:
                        output_ids = _gen_out.sequences
                        _past_kv = _gen_out.past_key_values
                    else:
                        output_ids = _gen_out
                        _past_kv = None
                t_generate = time.time() - t_generate

                # Detect whether generate() returned only the new tokens or the full
                # [input | output] sequence. Check by comparing the first prompt_len
                # tokens of output_ids against the input_ids — more reliable than a
                # length comparison (e.g. InternVL3-Instruct always returns new-tokens-
                # only, but its generations can be longer than prompt_len).
                if output_ids.shape[1] >= prompt_len:
                    output_only = not torch.equal(
                        output_ids[:, :prompt_len], inputs["input_ids"]
                    )
                else:
                    output_only = True  # output shorter than prompt → definitely new-only

                if output_only:
                    new_ids = output_ids
                    n_generated = output_ids.shape[1]
                else:
                    new_ids = output_ids[:, prompt_len:]
                    n_generated = new_ids.shape[1]

                decoded_new = processor.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                if output_only:
                    response = decoded_new
                    decoding_path = "new-tokens-only"
                else:
                    # For models returning the full sequence, strip the decoded prompt
                    # prefix to handle image-token expansion changing the sequence length.
                    decoded_prompt = processor.batch_decode(
                        inputs["input_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    if decoded_new.startswith(decoded_prompt):
                        response = decoded_new[len(decoded_prompt):]
                        decoding_path = "strip-prefix"
                    else:
                        response = decoded_new
                        decoding_path = "token-slice"

                if args.debug or not response:
                    log(f"  [debug sample={sample_id}] "
                        f"output_ids.shape={list(output_ids.shape)} "
                        f"prompt_len={prompt_len} n_generated={n_generated} "
                        f"output_only={output_only} decoding_path={decoding_path} "
                        f"response_len={len(response)}")
                    log(f"  [debug sample={sample_id}] new_ids (first 50 token IDs): {new_ids[0, :50].tolist()}")
                    if not response:
                        log(f"  [debug sample={sample_id}] new_ids decoded WITHOUT skip_special_tokens: "
                            f"{processor.batch_decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]!r}")
                        if not output_only:
                            log(f"  [debug sample={sample_id}] decoded_prompt (last 200 chars): {decoded_prompt[-200:]!r}")
                        log(f"  [debug sample={sample_id}] decoded_new (first 200 chars): {decoded_new[:200]!r}")

                if saliency_dir is not None:
                    if _sal_state is not None:
                        try:
                            sal, sal_tokens = _finalize_decode_saliency(
                                _sal_state, output_ids, _past_kv,
                                value_weighting=args.value_weighting,
                                head_reduction=args.head_reduction,
                                layer_reduction=args.layer_reduction,
                            )
                            sal_path = saliency_dir / f"{sample_id}.npz"
                            np.savez_compressed(sal_path, saliency=sal, tokens=np.array(sal_tokens))
                        except Exception as _sal_err:
                            log(f"  [saliency] sample={sample_id} failed: {_sal_err}")
                            sal_path = None
                        _past_kv = None  # free KV cache

                # Force an answer when generation stopped without a closed
                # <answer> block (max_new_tokens exhausted or a repetition loop).
                # Re-prompt with the original context + everything generated so
                # far + a forced "<answer>" opener (closing an open <think> first
                # so the glued response is parseable), then generate just the
                # answer and splice the two passes into one response.
                if args.force_answer and not re.search(
                    r"<answer>.*?</answer>", response, flags=re.DOTALL
                ):
                    forced_prefix = ""
                    if "<think>" in response and "</think>" not in response:
                        forced_prefix += "</think>\n"
                    forced_prefix += "<answer>"

                    # Trim trailing stop tokens so the continuation is not fed a
                    # mid-sequence EOS before the forced opener.
                    gen_ids = new_ids
                    while gen_ids.shape[1] > 0 and gen_ids[0, -1].item() in stop_token_ids:
                        gen_ids = gen_ids[:, :-1]

                    suffix_ids = torch.tensor(
                        [tokenizer.encode(forced_prefix, add_special_tokens=False)],
                        device=gen_ids.device, dtype=gen_ids.dtype,
                    )
                    forced_ids = torch.cat(
                        [inputs["input_ids"], gen_ids, suffix_ids], dim=1
                    )
                    forced_inputs = {
                        **inputs,
                        "input_ids": forced_ids,
                        "attention_mask": torch.ones(
                            forced_ids.shape, dtype=inputs["attention_mask"].dtype,
                            device=forced_ids.device,
                        ),
                    }
                    t_force = time.time()
                    with torch.inference_mode():
                        forced_out = model.generate(
                            **forced_inputs,
                            max_new_tokens=args.force_answer_max_new_tokens,
                            eos_token_id=stop_token_ids,
                        )
                    t_force = time.time() - t_force
                    t_generate += t_force

                    # generate() may return the full [input | output] sequence or
                    # only the new tokens; detect which (mirrors the main path).
                    if (forced_out.shape[1] >= forced_ids.shape[1]
                            and torch.equal(forced_out[:, :forced_ids.shape[1]], forced_ids)):
                        forced_new = forced_out[:, forced_ids.shape[1]:]
                    else:
                        forced_new = forced_out
                    forced_answer = processor.batch_decode(
                        forced_new,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    response = response + forced_prefix + forced_answer
                    if "</answer>" not in forced_answer:
                        response = response + "</answer>"

                    if args.debug:
                        log(f"  [debug sample={sample_id}] force-answer: "
                            f"added {forced_new.shape[1]} tokens in {t_force:.1f}s, "
                            f"forced_answer={forced_answer[:120]!r}")

            reasoning, final_answer = parse_response(response)
            correct = is_correct(final_answer, str(gt) if gt is not None else "", question)

            record = {
                "sample_id": sample_id,
                "image_filename": i,
                "question": question,
                "ground_truth": gt,
                "response": response,
                "reasoning": reasoning,
                "final_answer": final_answer,
                "correct": correct,
            }
            if saliency_dir is not None:
                # sal_path is set in the standard HF path; None for deepseek/internvl
                record["saliency_file"] = str(sal_path) if sal_path is not None else None
            out_f.write(json.dumps(record, default=str) + "\n")
            out_f.flush()

            n_correct_so_far += int(correct)
            n_done_so_far += 1
            elapsed = time.time() - t0
            if args.world_size > 1:
                if args.rank == 0 and _all_rank_outputs is not None:
                    g_done, g_correct = _aggregate_rank_progress(_all_rank_outputs)
                    g_acc = g_correct / g_done if g_done else 0.0
                    g_eta_min = (elapsed / g_done) * (n_total_global - g_done) / 60 if g_done else 0.0
                    log(f"  [{g_done}/{n_total_global}] ({100*g_done/n_total_global:.1f}%) "
                        f"acc={g_acc:.3f} eta={g_eta_min:.1f}min")
            else:
                n_done_this_run = n_done_so_far - len(done_ids)
                avg = elapsed / n_done_this_run if n_done_this_run > 0 else 0
                eta_min = avg * (len(ds) - i - 1) / 60
                tok_per_sec = n_generated / t_generate if t_generate > 0 else 0
                log(f"  [{i + 1}/{n_total_global}] correct={correct} "
                    f"acc={n_correct_so_far/n_done_so_far:.3f} "
                    f"prompt_tok={prompt_len} gen_tok={n_generated} "
                    f"preprocess={t_preprocess:.1f}s generate={t_generate:.1f}s "
                    f"({tok_per_sec:.1f} tok/s) "
                    f"avg={avg:.1f}s/it eta={eta_min:.1f}min")

    # Compute final accuracy by reading all completed records.
    all_results = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))
    n_correct = sum(1 for r in all_results if r["correct"])
    log(f"Saved {len(all_results)} results to {out_path.resolve()}")
    if all_results:
        log(f"Final accuracy: {n_correct}/{len(all_results)} "
            f"= {n_correct/len(all_results):.4f}")

    # The landing check. A VGA run whose injection never fired is a run of the
    # stock model under a guided run's filename, and nothing else in the output
    # would say so.
    if vga is not None:
        d = vga.diagnostics()
        log(f"VGA: guided {d['guided']}/{d['calls']} generate calls "
            f"({d['no_image']} had no image), {d['object_mode']} object-directed / "
            f"{d['agnostic_mode']} object-agnostic ({d['fallbacks']} fell back), "
            f"{d['injected_steps']} decode steps injected on layers {d['layers']}, "
            f"mean relative update {d['mean_rel_update']:.4f}, beta_eff {d['last_beta_eff']:.4f}")
        if d["injected_steps"] == 0:
            log("VGA: WARNING — nothing was ever injected; these are stock-model scores.")

    if args.world_size > 1 and _all_rank_outputs is not None and _done_counter_path is not None:
        n_ranks_done = _increment_file_counter(_done_counter_path)
        log(f"Rank {args.rank} done ({n_ranks_done}/{args.world_size} ranks finished)")
        if n_ranks_done >= args.world_size:
            merged_path = out_path.parent / (_base_stem + out_path.suffix)
            existing = [p for p in _all_rank_outputs if p.exists()]
            n_merged = merge_rank_files(existing, merged_path)
            with merged_path.open() as f:
                merged = [json.loads(l) for l in f if l.strip()]
            n_merged_correct = sum(1 for r in merged if r.get("correct"))
            log(f"Merged {len(existing)}/{args.world_size} shard(s) → {n_merged} records → {merged_path}")
            if n_merged:
                log(f"Merged accuracy: {n_merged_correct}/{n_merged} = {n_merged_correct/n_merged:.4f}")
            for p in existing:
                p.unlink()
                log(f"Deleted rank shard: {p}")
            for _cp in (_counter_path, _done_counter_path, _counter_ready_path):
                try:
                    _cp.unlink()
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
