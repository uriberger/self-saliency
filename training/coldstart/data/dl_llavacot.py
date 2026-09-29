import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

from selfsal.data.paths import DATA_ROOT

COLDSTART = DATA_ROOT / "coldstart"   # override the root with SELFSAL_DATA


t0 = time.time()
p = snapshot_download(
    "Xkev/LLaVA-CoT-100k", repo_type="dataset",
    local_dir=str(COLDSTART / "LLaVA-CoT-100k"),
    allow_patterns=["image.zip.part-*", "train.jsonl"],
    max_workers=8)
print("DONE", p, "in", round((time.time() - t0) / 60, 1), "min")
