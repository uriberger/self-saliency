# Copyright 2026 NVIDIA. Apache-2.0.
"""Localising the objects a reasoning step mentions (Section 3.3).

Grounding-DINO over the step's whole sentence, returning boxes in relative
[x0, y0, x1, y1] coordinates. `grounding.mask` turns those into the region u_s.

TWO BACKENDS, ONE INTERFACE. `ground()` calls a local detector by default and a served
one when `api_base` is set. The served path exists because the trainer's layout puts the
detector on a GPU of its own beside vLLM, outside the policy's allocation; the local
path is what every analysis script uses. The served path falls back to local on any
error rather than letting a sidecar hiccup end a 46-hour run.

RELATIVE COORDINATES, ALWAYS. Boxes come back as fractions of the image, never pixels.
That is what lets the same box be rasterised onto whatever patch grid the model happened
to choose for that image -- Qwen3-VL's grid is image-dependent -- and it is why the
Section 5 corpus can be rebuilt at a different resolution without re-grounding.

THRESHOLD. `box_threshold` is applied here and area filtering is not: the caller owns
the area caps, so they stay run-time knobs over stored boxes. The paper's runs use 0.10
for both the box and text thresholds.
"""

from __future__ import annotations

import base64
import contextlib
import io

GROUNDING_DINO_HF_ID = "IDEA-Research/grounding-dino-base"

DEFAULT_BOX_THRESHOLD = 0.10
DEFAULT_BATCH_SIZE = 32

#: Lazily-loaded local detector, one per process.
_LOCAL: dict = {"proc": None, "model": None, "device": None, "hf_id": None}


@contextlib.contextmanager
def _no_deepspeed_zero3_init():
    """Hide HF's global ZeRO-3 config from `from_pretrained` for the duration.

    Same hazard as the step classifier's: under ZeRO-3 every later `from_pretrained` is
    wrapped in `deepspeed.zero.Init` and has its parameters 1-D sharded, and a sharded
    weight is no longer 2-D at forward. This detector is small, frozen and
    single-device, and must be fully materialised.
    """
    try:
        import transformers.integrations.deepspeed as ds
    except Exception:
        yield
        return
    saved = getattr(ds, "_hf_deepspeed_config_weak_ref", None)
    ds._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        ds._hf_deepspeed_config_weak_ref = saved


def load_local(hf_id: str = GROUNDING_DINO_HF_ID, device: str | None = None):
    """(processor, model, device) for the in-process detector, loaded once."""
    if _LOCAL["model"] is None or _LOCAL["hf_id"] != hf_id:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(hf_id)
        with _no_deepspeed_zero3_init():
            model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id)
            model = model.to(device).eval()
        _LOCAL.update(proc=proc, model=model, device=device, hf_id=hf_id)
    return _LOCAL["proc"], _LOCAL["model"], _LOCAL["device"]


def _as_prompt(text: str) -> str:
    """Grounding-DINO expects a period-terminated phrase list."""
    t = text.strip()
    return t if t.endswith(".") else t + "."


def ground_local(images, texts, box_threshold: float = DEFAULT_BOX_THRESHOLD,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 hf_id: str = GROUNDING_DINO_HF_ID, device: str | None = None):
    """Batched local detection. -> one list of relative boxes per (image, text) pair.

    HALVES THE BATCH ON OOM RATHER THAN DYING. Deformable attention materialises one
    contiguous (batch, queries, heads, levels, points) tensor, so peak memory scales with
    the batch AND with the images' native resolution -- a batch that fits for one caller
    OOMs for the next. Callers that share a GPU with an 8B policy cannot pick a size that
    is both safe and fast. A single item that still OOMs is a real failure and is
    re-raised.
    """
    import torch

    proc, model, device = load_local(hf_id, device)
    prompts = [_as_prompt(t) for t in texts]
    out: list = [None] * len(images)

    def run_chunk(start: int, n: int):
        imgs = images[start:start + n]
        inputs = proc(images=imgs, text=prompts[start:start + n], return_tensors="pt",
                      padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=float(box_threshold),
            text_threshold=float(box_threshold),
            target_sizes=[(im.size[1], im.size[0]) for im in imgs])   # (h, w)
        for j, res in enumerate(results):
            w, h = imgs[j].size
            out[start + j] = [[x1 / w, y1 / h, x2 / w, y2 / h]
                              for x1, y1, x2, y2 in res["boxes"].tolist()]

    start, size = 0, int(batch_size)
    while start < len(images):
        n = min(size, len(images) - start)
        while True:
            try:
                run_chunk(start, n)
                break
            except torch.cuda.OutOfMemoryError:
                if n == 1:
                    raise
                torch.cuda.empty_cache()
                n = max(1, n // 2)
                print(f"[dino] CUDA OOM; retrying at offset {start} with batch {n}",
                      flush=True)
        start += n
    return out


def ground_served(images, texts, api_base: str,
                  box_threshold: float = DEFAULT_BOX_THRESHOLD, timeout: int = 120):
    """The same call against a `grounding.server` endpoint."""
    import requests

    payload = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        payload.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    resp = requests.post(
        api_base.rstrip("/") + "/ground",
        json={"images": payload, "texts": list(texts),
              "box_threshold": float(box_threshold),
              "text_threshold": float(box_threshold)},
        timeout=timeout)
    resp.raise_for_status()
    return resp.json()["boxes"]


def ground(images, texts, box_threshold: float = DEFAULT_BOX_THRESHOLD,
           api_base: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE,
           hf_id: str = GROUNDING_DINO_HF_ID, device: str | None = None):
    """Ground each text against its image. -> one list of relative boxes per pair.

    Falls back from the served detector to a local one on any error: a reward-server
    hiccup should cost a step's latency, not the run.
    """
    if not images:
        return []
    if api_base:
        try:
            return ground_served(images, texts, api_base, box_threshold)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[dino] served detector failed ({exc}); falling back to local",
                  flush=True)
    return ground_local(images, texts, box_threshold=box_threshold,
                        batch_size=batch_size, hf_id=hf_id, device=device)


def ground_scored(images, texts, box_threshold: float = DEFAULT_BOX_THRESHOLD,
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  hf_id: str = GROUNDING_DINO_HF_ID, device: str | None = None):
    """Like `ground_local`, but keeps each box's confidence.

    -> one list of (box, score) per (image, text) pair.

    The head-selection screen needs the scores because it stores boxes once at a LOW
    threshold and then filters at several higher ones offline -- re-running the detector
    per threshold over Visual-CoT would be the expensive half of the screen repeated for
    nothing. The reward path does not need them: it grounds at one threshold and the
    detector has already applied it.
    """
    import torch

    proc, model, device = load_local(hf_id, device)
    prompts = [_as_prompt(t) for t in texts]
    out: list = [None] * len(images)

    def run_chunk(start: int, n: int):
        imgs = images[start:start + n]
        inputs = proc(images=imgs, text=prompts[start:start + n], return_tensors="pt",
                      padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=float(box_threshold), text_threshold=float(box_threshold),
            target_sizes=[(im.size[1], im.size[0]) for im in imgs])
        for j, res in enumerate(results):
            w, h = imgs[j].size
            out[start + j] = [
                ([x1 / w, y1 / h, x2 / w, y2 / h], float(score))
                for (x1, y1, x2, y2), score in zip(res["boxes"].tolist(),
                                                   res["scores"].tolist())]

    start, size = 0, int(batch_size)
    while start < len(images):
        n = min(size, len(images) - start)
        while True:
            try:
                run_chunk(start, n)
                break
            except torch.cuda.OutOfMemoryError:
                if n == 1:
                    raise
                torch.cuda.empty_cache()
                n = max(1, n // 2)
        start += n
    return out


def ground_claim(image, text, box_threshold: float = DEFAULT_BOX_THRESHOLD, **kw):
    """One image, one phrase, scored. -> [(box, score), ...]."""
    return ground_scored([image], [text], box_threshold=box_threshold, **kw)[0]
