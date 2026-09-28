# Copyright 2026 NVIDIA. Apache-2.0.
"""Grounding-DINO as an HTTP sidecar, for the GRPO reward.

The trainer's layout puts the detector on a GPU of its own beside vLLM, outside the
policy's ZeRO-3 allocation, and every (observe-step image, step-text) pair of a rollout
arrives as one request per reward call.

    python -m selfsal.grounding.server --port 8100
    # then train with:  regions.api_base: http://<node>:8100

    POST /ground
      {"images": [<base64 PNG>, ...], "texts": [...],
       "box_threshold": float, "text_threshold": float}
      -> {"boxes": [[[x0, y0, x1, y1], ...], ...]}   per item, RELATIVE coordinates
    GET  /health  -> {"status": "ok", "model": ..., "device": ...}

SERVED AND LOCAL ARE THE SAME CODE, not two implementations that agree. The detection
itself is `dino.ground_local`, so a run that uses the sidecar and a run that loads the
detector in-process cannot drift apart -- which they could while this was a second
transcription of the same twenty lines.

NO AREA FILTERING HAPPENS HERE, deliberately. Both caps are the caller's: `max_box_area`
is per box, and `max_union_area` is per step on the rasterised union, which needs the
patch grid the server has never seen. The server returns everything above
`box_threshold` and `grounding.mask.union_mask` decides what survives -- so changing
either cap never means restarting this process.
"""

from __future__ import annotations

import argparse
import base64
import io
import os

from PIL import Image

from .dino import GROUNDING_DINO_HF_ID, ground_local, load_local

#: Per-forward batch cap. Grounding-DINO-base (Swin-B + a deformable encoder) with the
#: processor's ~800x1333 resize and padding-to-max needs several GB per image, so 32 can
#: OOM an 80 GB card. The endpoint still accepts an arbitrarily long request and chunks
#: it internally; `dino.ground_local` halves again on OOM beneath this.
SERVER_BATCH = int(os.environ.get("SELFSAL_DINO_SERVER_BATCH", "8"))


def _decode(b64: str) -> Image.Image:
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    return im if im.mode == "RGB" else im.convert("RGB")


def build_app():
    """Construct the FastAPI app. Deferred so importing this module needs no web stack."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    class GroundRequest(BaseModel):
        images: list[str]                 # base64-encoded PNG bytes
        texts: list[str]
        box_threshold: float = 0.10
        text_threshold: float = 0.10      # accepted and kept equal; see below

    class GroundResponse(BaseModel):
        boxes: list[list[list[float]]]

    app = FastAPI(title="selfsal-grounding-dino")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": GROUNDING_DINO_HF_ID,
                "device": load_local()[2]}

    @app.post("/ground", response_model=GroundResponse)
    def ground(req: GroundRequest):
        if not req.images or len(req.images) != len(req.texts):
            return {"boxes": []}
        images = [_decode(b) for b in req.images]
        # The paper's runs set both thresholds to the same value and `ground_local`
        # takes one. A request that disagrees is a configuration the trained arms never
        # used, so it is refused rather than silently resolved to one of the two.
        if abs(float(req.box_threshold) - float(req.text_threshold)) > 1e-9:
            raise ValueError(
                f"box_threshold {req.box_threshold} != text_threshold "
                f"{req.text_threshold}; the trained arms used a single value for both")
        boxes = ground_local(images, req.texts,
                             box_threshold=float(req.box_threshold),
                             batch_size=SERVER_BATCH)
        return {"boxes": boxes}

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve Grounding-DINO for the GRPO reward.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args(argv)

    import uvicorn

    app = build_app()
    load_local()      # eager, so /health is meaningful at once and request 1 is not slow
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
