# `gh-pages` — the Self-Saliency project page

This branch is **only** the public project page. It is an orphan branch: it
shares no history with `main`, and none of the research code, tests or
evaluation results are on it. `main` keeps its `docs/` directory for the
developer documentation; nothing here collides with it.

```
index.html            the whole page
.nojekyll             serve the files as-is; do not run Jekyll over them
static/css/index.css  design tokens + layout
static/js/index.js    theme toggle, motion-respecting teaser playback
static/images/        figures 1-5 (WebP, from the paper PDF), poster, social card
static/videos/        the teaser clip
```

Total payload is about 1.7 MB.

## Publishing it

1. Make the repository public.
2. **Settings → Pages → Source: _Deploy from a branch_ → `gh-pages` / `/` (root)**.
3. Set the repository's *Website* field to `https://uriberger.github.io/self-saliency/`.

Note that GitHub Pages on a *private* repository needs a paid plan; on the free
plan the site only builds once the repository is public.

## Before it goes live

`index.html` opens with a yellow placeholder banner listing everything still
unresolved. Every one of those is also marked inline with `<mark class="todo">`
or a `TODO(publish)` comment, so nothing ships silently:

- **arXiv link and ID** — the link button, the `arXiv:` field of the BibTeX
  entry, and the `citation_arxiv_id` / `citation_pdf_url` meta tags.
- **Hugging Face links** for the released checkpoints (one repository per arm —
  see `docs/publishing.md` on `main`).

Authors are filled in. They carry no homepage links; add them by wrapping each
name in the `.authors` paragraph in an `<a>`.

Delete the banner element once they are all done.

## Where the assets came from

Figures 1–5 are rendered straight from `self_saliency_final.pdf` at ~1800 px
wide and converted to WebP. Regenerating them means re-running the extraction
against the current PDF — they are not hand-edited.

The teaser is `fig1_steps_video.py` in the research repository, on
CV-Bench row 167:

```
--run-dir outputs/saliency_viz/fig1d-search --model ours \
--sample sample_167_row000167 \
--smooth 1.0 --upsample map --overlay-mode alpha --alpha 0.8 \
--map-label "" \
--question "Estimate the real-world distances between objects in this image. \
Which object is closer to the chair (red box), the bookcase (blue box) or the \
table (green box)? (A) bookcase (B) table"
```

`--map-label ""` drops the `step k of n · glimpse` suffix; `glimpse` is an
internal saliency-map name and means nothing to a reader. The overlay knobs
were not recorded anywhere and were recovered by sweeping against an earlier
render until a frame matched pixel-for-pixel.

That command reproduces this clip as long as the script is at `ec94522` or
later. Frames are 1346×716; an earlier render of the same sample was 694 tall,
because `355c4ef` reserved a footer line and centred the chain against the
picture. `--smooth` is cosmetic and its sigma is in patches, so it does not
carry over to a sample on a different grid.

## Editing

It is one static HTML file with no build step and no dependencies. Open
`index.html` in a browser and reload.

The design tokens at the top of `index.css` (surfaces, ink roles, the blue
accent) come from a validated palette, with the dark mode stepped for the dark
surface rather than flipped from the light one. Both modes are deliberate; if
you change one, change the other.
