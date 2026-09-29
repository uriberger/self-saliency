"""Robustly split a Saliency-R1 cold JSON array into records by the per-record start
marker `{"image": "`, validating each record independently. Isolates malformed records
(e.g. unescaped quotes) instead of letting one poison the whole file. Writes clean JSONL."""
import json, sys, re

src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
clean = bytes((b if b >= 0x20 else 0x20) for b in raw).decode("utf-8", errors="replace")
clean = clean.strip()
assert clean[0] == "[" and clean[-1] == "]", (clean[:5], clean[-5:])
body = clean[1:-1].strip()

MARK = '{"image": "'
# indices where each record begins
idxs = [m.start() for m in re.finditer(re.escape(MARK), body)]
assert idxs and idxs[0] == 0, idxs[:3]
ok = bad = 0
bad_samples = []
with open(dst, "w", encoding="utf-8") as f:
    for i, start in enumerate(idxs):
        end = idxs[i + 1] if i + 1 < len(idxs) else len(body)
        chunk = body[start:end].rstrip().rstrip(",").rstrip()
        try:
            rec = json.loads(chunk)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
        except Exception as e:
            bad += 1
            if len(bad_samples) < 5:
                bad_samples.append((i, str(e)[:80], chunk[:100]))
print(f"records: {len(idxs)}  ok: {ok}  bad: {bad}")
for b in bad_samples:
    print("  BAD idx", b[0], "|", b[1], "|", b[2])
