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

## Before it goes live

`index.html` opens with a yellow placeholder banner listing everything still
unresolved. Every one of those is also marked inline with `<mark class="todo">`
or a `TODO(publish)` comment, so nothing ships silently:

- **arXiv link and ID** — the link button, the `arXiv:` field of the BibTeX
  entry, and the `citation_arxiv_id` / `citation_pdf_url` meta tags.
- **The code repository.** The Code button is deliberately dead and captioned
  "Coming soon" until `main` is public; the footer does not link it either, so
  the URL is not advertised anywhere on the page.
- **Hugging Face links** for the released checkpoints (one repository per arm —
  see `docs/publishing.md` on `main`).

Authors are filled in. They carry no homepage links; add them by wrapping each
name in the `.authors` paragraph in an `<a>`.

Delete the banner element once they are all done.

## Where the assets came from

Figures 2–5 are rendered straight from `self_saliency_final.pdf` at ~1800 px
wide and converted to WebP. Regenerating them means re-running the extraction
against the current PDF — they are not hand-edited. Paper Figure 1 is no longer
shown on the page (the teaser carries that example instead), but
`og-image.jpg`, the social card, is still rendered from it.

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
