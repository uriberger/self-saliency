"""Convert a Saliency-R1 cold JSON (single-line array, with stray unescaped control
chars in some strings) into clean JSONL that HF datasets/pyarrow reads reliably."""
import json, sys, re

src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
# Replace raw control bytes 0x00-0x1f (single-byte ASCII, never part of a multibyte
# UTF-8 sequence) with a space. These are illegal unescaped inside JSON strings.
clean = bytes((b if b >= 0x20 or b in (0x20,) else 0x20) for b in raw)
text = clean.decode("utf-8", errors="replace")
data = json.loads(text)
assert isinstance(data, list), type(data)
with open(dst, "w", encoding="utf-8") as f:
    for rec in data:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"OK {src} -> {dst}: {len(data)} records")
