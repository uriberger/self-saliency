# Copyright 2026 NVIDIA. Apache-2.0.
"""Shard progress with an ETA, and the monitor that aggregates shards into one line.

Every GPU stage in `experiments/` is embarrassingly parallel over its unit -- a picture,
a completion, an (image, sentence) pair -- so it runs as N shards that each write a
heartbeat file, and one monitor process reads them all. No IPC, and a dead shard costs
only its own slice.

Extracted from the archive's `intervene_probe.py`, an attention-intervention experiment
that is not part of this paper: the probes borrowed exactly two names from it, `Progress`
and `monitor`, so those come across and the intervention stays behind.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


def fmt_dt(s):
    if s is None or not math.isfinite(s):
        return "?"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """Rolling-rate progress with an ETA, to stdout and to a heartbeat JSON.

    The rate is an EWMA rather than a simple average: the first units of a shard
    include the model load and are not representative of the steady state.
    """

    HEARTBEAT_SECS = 15.0

    def __init__(self, path: Path, total: int, label: str, log_every: int = 25,
                 already_done: int = 0):
        self.path, self.total, self.label = path, int(total), label
        self.log_every = max(1, int(log_every))
        self.done, self.resumed, self.rate = 0, int(already_done), None
        self.t0 = self.tlast = self.twrite = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        print(f"[{label}] starting: {self.resumed} already done of {self.total}", flush=True)

    @property
    def completed(self):
        return self.resumed + self.done

    def tick(self, n: int = 1):
        now = time.time()
        dt = (now - self.tlast) / max(1, n)
        self.tlast = now
        self.rate = dt if self.rate is None else 0.9 * self.rate + 0.1 * dt
        self.done += n
        # Count-based cadence alone makes a slow stage look dead: `prepare` runs at
        # ~7 s/sample, so log_every=25 is a heartbeat every three minutes and the
        # monitor reported 0/8 for an entire smoke run that was working fine. Write
        # on elapsed time as well, so the monitor's picture is never more than
        # HEARTBEAT_SECS stale whatever the per-unit cost of the stage.
        if (self.done % self.log_every == 0 or self.completed >= self.total
                or now - self.twrite >= self.HEARTBEAT_SECS):
            self.twrite = now
            self._write()
            print(self.line(), flush=True)

    def eta_seconds(self):
        if not self.rate or self.completed >= self.total:
            return 0.0
        return (self.total - self.completed) * self.rate

    def line(self):
        pct = 100.0 * self.completed / max(1, self.total)
        rate = (1.0 / self.rate) if self.rate else 0.0
        return (f"[{self.label}] {self.completed}/{self.total} ({pct:5.1f}%)  "
                f"{rate:.2f} it/s  elapsed {fmt_dt(time.time() - self.t0)}  "
                f"ETA {fmt_dt(self.eta_seconds())}")

    def _write(self):
        payload = {"label": self.label, "total": self.total, "completed": self.completed,
                   "rate": self.rate, "eta": self.eta_seconds(), "updated": time.time(),
                   "pid": os.getpid()}
        try:                                  # a heartbeat is never worth killing a run
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.path)
        except OSError:
            pass

    def close(self):
        self._write()
        print(self.line(), flush=True)


def monitor(out_dir: Path, interval: float, once: bool = False, stage: str = ""):
    """Aggregate every shard heartbeat into one progress line + ETA.

    `once` prints a single line and returns; the launcher uses it after `wait` so the
    final state is shown, which the polling loop otherwise misses (it gets killed the
    moment the shards exit).
    """
    beats = out_dir / "progress"
    if not once:
        print(f"[monitor] watching {beats} (ctrl-C to stop)", flush=True)
    while True:
        entries = []
        # Heartbeats are named <stage><shard>.json. Without the filter the monitor
        # sums a finished `prepare` into a running `run` and reports 100% at once.
        pattern = f"{stage}*.json" if stage else "*.json"
        for f in sorted(beats.glob(pattern)) if beats.is_dir() else []:
            try:
                entries.append(json.loads(f.read_text()))
            except Exception:
                continue                       # mid-write or torn: skip this round
        if not entries:
            print("[monitor] no heartbeats yet", flush=True)
            if once:
                return
            time.sleep(interval)
            continue
        now = time.time()
        tot = sum(e["total"] for e in entries)
        comp = sum(e["completed"] for e in entries)
        alive = [e for e in entries if now - e.get("updated", 0) <= 600]
        rates = [1.0 / e["rate"] for e in alive if e.get("rate")]
        # Shards run in parallel, so wall-clock ETA is the SLOWEST shard's, not the sum.
        etas = [(e["total"] - e["completed"]) * e["rate"] for e in alive if e.get("rate")]
        print(f"[monitor] {comp}/{tot} ({100.0 * comp / max(1, tot):5.1f}%)  "
              f"{sum(rates):.1f} it/s total  "
              f"{len(alive)}/{len(entries)} shards alive  "
              f"ETA {fmt_dt(max(etas) if etas else None)}", flush=True)
        if once:
            return
        if comp >= tot:
            print("[monitor] all shards complete", flush=True)
            return
        if not alive:
            # Every heartbeat has gone stale with work outstanding: the shards died.
            # Return rather than spin forever, so the launcher's foreground monitor
            # never wedges an interactive shell.
            print("[monitor] no shard has reported in 10 min and work remains -- "
                  "check the shard logs; re-running the same command resumes",
                  flush=True)
            return
        time.sleep(interval)


