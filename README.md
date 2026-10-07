# `gh-pages` — the Self-Saliency project page

This branch is **only** the public project page. It is an orphan branch: it
shares no history with `main`, and none of the research code, tests or
evaluation results are on it. `main` keeps its `docs/` directory for the
developer documentation; nothing here collides with it.

```
index.html                     the whole page
.nojekyll                      serve the files as-is; do not run Jekyll over them
static/css/nvidia-project.css  the shared NVIDIA Research project stylesheet, vendored
static/css/index.css           what that file does not cover
static/js/index.js             motion-respecting teaser playback
static/images/                 figures 2-5 (WebP, from the paper PDF), poster, social card
static/videos/                 the teaser clip
```

Total payload is about 3.5 MB, nearly all of it the teaser.

## Publishing it

1. Make the repository public.
2. **Settings → Pages → Source: _Deploy from a branch_ → `gh-pages` / `/` (root)**.
3. Set the repository's *Website* field to `https://uriberger.github.io/self-saliency/`.

Note that GitHub Pages on a *private* repository needs a paid plan; on the free
plan the site only builds once the repository is public.

## Still to land

The page is meant to be live as it stands; the placeholder banner is gone and
nothing on it is a stub. Two artefacts are not out yet, and each is represented
by a dead button captioned "Coming soon" rather than by a link that 404s:

- **The code repository.** The Code button is inert and the footer does not link
  it either, so the URL is not advertised anywhere on the page. To release:
  make `main` public, point the button at it and drop the caption.
- **The weights.** A Hugging Face repository per arm — see `docs/publishing.md`
  on `main`. Same change: give the button an `href` and remove its caption.

Each dead button also carries a plain-English `title`, because that attribute
is a tooltip a visitor can read. Never park a note to yourself in one.

The paper is on arXiv as
[2610.05023](https://arxiv.org/abs/2610.05023) (v1, 2026-10-04). The Paper
button, the BibTeX entry and the `citation_*` meta tags all carry that id; if a
v2 goes up, none of them need to change — arXiv's unversioned URLs resolve to
the latest.

Gal Chechik and Gal Dalal link to their `research.nvidia.com/person/` pages;
Uri Berger has none (that URL 404s), so the green is set on the paragraph
rather than per-name. Add a link by wrapping the name in an `<a>`.

## Where the assets came from

Figures 3 and 5 are rendered straight from `self_saliency_final.pdf` at ~1800 px
wide and converted to WebP. Regenerating them means re-running the extraction
against the current PDF — they are not hand-edited. Paper Figure 1 is no longer
shown on the page (the teaser carries that example instead), but
`og-image.jpg`, the social card, is still rendered from it.

Three are not straight from the PDF:

- **`fig2.webp`** is `pipeline_with_rewards.png` from the research repository,
  not the paper's Figure 2. It is the same pipeline with the three reward terms
  and their path into GRPO drawn on.
- **`fig4.webp`** is the paper's Figure 4 with a vertical rule added at
  x&nbsp;=&nbsp;1400 of the 1801 px render, in the gutter between `nemotron` and
  `human boxes`, to separate the four models from the human annotation. The
  gutter was found by scanning for columns that are white across the plot band,
  so the rule sits between panels rather than over one. Nothing else in the
  figure is touched. Re-extracting Figure 4 from the PDF drops the rule; re-run
  that step if you do.
- **`fig5.webp`** is regenerated from the scans, not taken from the paper at
  all. `sink_location_xmodel_tables.py --panels` draws the two arms
  (`box_qwen3vl_auto` and `box_overlap_wov04_auto`, generated tokens, L22
  h28/31) and `fig5_overrange_compose.py` joins them; both live in the research
  repository. The colour ramp stops at 4x, so a 22x corner and a 12x corner were
  both "saturated red" and the figure said they were equal — patches past 16x
  now carry their own colour and their own key. The per-panel subtitles and the
  provenance note are cropped for the web copy: both are illegible at the size
  this renders, and both now sit in the figcaption instead.

  **The numbers differ from the paper.** This pair reads 22 → 12; the published
  Figure 5 reads 51 → 12. The *after* panel matches the scan exactly (11.51
  renders as "12", its top-right 1.41 as "1.4"); the *before* panel does not —
  the scan gives 22.22 and 0.75 against the published 51 and 1.1 — and no
  `tables.md` under `outputs/sink_location/` holds a top-left anywhere in 46–56.
  The page shows the regenerated pair because both of its panels come from one
  scan. Whether the paper's 51 is right is an open question for the paper.

The teaser is the research repository's
`outputs/fig1-multistep/video-count-soccer/landscape/chain.mp4`: two counting
questions, vanilla and Self-Saliency side by side, concatenated — see
`concat.json` beside it for the two source clips and their samples.

It is **not** committed as rendered. The source is 2066×1586 and 6.8 MB, which
is too heavy to put in front of a reader, so it is transcoded to 1600×1228 with
`libx264 -crf 25 -preset slow -tune stillimage +faststart` (2.8 MB). `tune
stillimage` is the right one here: the clip is held frames with dissolves
between them, not motion. Re-run that transcode rather than editing the file in
place.

The video's own canvas is `#111111`; `.fig--dark` in the stylesheet matches it
so the clip does not sit inside a lighter box. If the clip is re-rendered on a
different background, change that value too.

## Editing

It is one static HTML file with no build step and no dependencies. Open
`index.html` in a browser and reload.

### The styling

The page follows the NVIDIA Research project-page template, the one
[PlaMo](https://research.nvidia.com/labs/par/projects/plamo/) uses.
`static/css/nvidia-project.css` is NVIDIA's own shared stylesheet, vendored
byte-for-byte from
`research.nvidia.com/labs/par/projects/assets/nvidia-project.css`. It supplies
the top bar, the green section rules, `.nv-btn`, `.nv-bibtex`, the footer and
every `--nv-*` token. It is **copied rather than hot-linked**: a GitHub Pages
site should not depend on another origin's asset for its whole appearance, and
hot-linking would break the page silently if NVIDIA moved it. If this page ever
moves under `research.nvidia.com`, delete the copy and link the shared asset.

`index.css` is only the parts that file does not cover: the paper head, the
tables, the teaser canvas, the placeholder banner.

Two things to know before editing:

- **It is light only.** The NVIDIA template has no dark variant, so the theme
  toggle that used to be here is gone. Adding one back means picking dark steps
  for every `--nv-*` token, not flipping them.
- **The shared file's `.nv-authors a` is white with a grey underline** — it is
  written for `.nv-hero`, the black hero block this page does not use. On the
  white paper head that renders author names invisible above their underlines.
  `index.css` overrides it. Expect the same trap from any other `.nv-hero`
  descendant rule you reuse outside that block.

The top bar links to NVIDIA Research, the PAR Lab and its project index, with
absolute URLs because this page is not served from that site.
