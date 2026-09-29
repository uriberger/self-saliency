"""Extract ONLY the referenced LLaVA-CoT images from image.zip, in parallel.
The zip holds the full 1.39M-file pool; the cold-start JSON needs 50,411 unique files.
Parallel process workers each open the zip read-only and extract their chunk."""
import os, sys, zipfile, time
from concurrent.futures import ProcessPoolExecutor

from selfsal.data.paths import DATA_ROOT

COLDSTART = DATA_ROOT / "coldstart"   # override the root with SELFSAL_DATA


ZIP = str(COLDSTART / "LLaVA-CoT-100k/image.zip")
DST = str(COLDSTART / "Saliency-R1-cold/llava_cot_images")
LISTFILE = "/tmp/needed_entries.txt"
NWORK = 12

names = [l.rstrip("\n") for l in open(LISTFILE)]
os.makedirs(DST, exist_ok=True)

def worker(chunk):
    ok = miss = 0
    with zipfile.ZipFile(ZIP) as zf:
        have = zf.NameToInfo
        for n in chunk:
            info = have.get(n)
            if info is None:
                miss += 1
                continue
            # skip if already extracted (resume-safe)
            dst = os.path.join(DST, n)
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                ok += 1
                continue
            zf.extract(info, DST)
            ok += 1
    return ok, miss

chunks = [names[i::NWORK] for i in range(NWORK)]
t0 = time.time()
tot_ok = tot_miss = 0
with ProcessPoolExecutor(max_workers=NWORK) as ex:
    for ok, miss in ex.map(worker, chunks):
        tot_ok += ok; tot_miss += miss
print(f"EXTRACT DONE: {tot_ok} extracted, {tot_miss} missing, in {round((time.time()-t0)/60,1)} min")
