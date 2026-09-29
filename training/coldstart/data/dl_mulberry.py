import os
os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

from selfsal.data.paths import DATA_ROOT

COLDSTART = DATA_ROOT / "coldstart"   # override the root with SELFSAL_DATA

p = snapshot_download(
    "HuanjinYao/Mulberry-SFT", repo_type="dataset",
    local_dir=str(COLDSTART / "Mulberry-SFT"),
    allow_patterns=["mulberry_images.tar"], max_workers=8)
print("DONE", p)
