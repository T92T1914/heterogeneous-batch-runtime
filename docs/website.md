# Runtime evidence page

The static page explains the retained CPU and CUDA collections without running a
numerical workload or downloading a model. It keeps the original CUDA losses,
small-input CPU wins, session setup costs and faster direct CPU inference visible.
Visitors can compare a condition, choose a complete-call or CUDA-event interval,
and follow the result into its exact raw record, protocol and executed source.

Serve `web/` as the website directory. It has no production JavaScript dependencies,
external fonts, analytics, runtime APIs, model weights or provider binaries. The
full comparison tables, inference outputs and memory observations remain available
without JavaScript. Wide tables scroll within labelled keyboard-focusable regions.

## Data and regeneration

`web/source-pins.json` identifies the 22 retained input files from source
`34e2b1151681ba231eb38713bde1a2f21cf8444f`, including the separate pinned theme
adapter. The generator rejects changed input bytes before calculating a table.
It derives medians, inclusive quartiles, minima, maxima and first invocations from
the raw CSV/JSON samples. It never imports a native extension, starts inference,
acquires an artifact or calls a benchmark. Historical measurements retain their
executed source identities. Adding a page does not rerun them at a later revision.

```sh
python tools/build_site.py --write
python tools/build_site.py
python -m unittest discover -s tests -p test_site.py -v
node --test tests/site-selection.test.mjs
cd tests/browser
npm ci --ignore-scripts
npx playwright install chromium webkit firefox
npm test
```

`--write` updates only `web/index.html`, `web/evidence.json` and
`web/appearance.css`. The command without `--write` checks their exact bytes.
Authored prose is in `web/index.template.html`. Both the embedded data and
downloadable JSON contain the same facts. New evidence needs a deliberate input
pin and semantic review, rather than accepting a changed record automatically.

Six collections remain separate: initial CPU paths, original CPU/CUDA comparison,
CPU stencil dispatch follow-up, histogram reuse, CPU inference and required CUDA
inference. The original native boundary includes result disposal. The histogram
reuse boundary ends with an owned host result and excludes its later destruction.
Inference timings start with prepared tensors and exclude PNG preparation. Device
events are not added to host intervals. ONNX operator placement is not kernel time.

## Appearance and accessibility

The thin CSS adapter uses `clair-obscur-themes/tokens.json` at
`7a57fe750ff50205a17e1d342106a0d3f2777159`. The exact copied token bytes have SHA-256
`889df65e0f4f4eea99a81c535d542e7332b537c000c332402a3ffa6a6cbb15b7`.
Auto follows system appearance. An explicit Clair or Obscur selection is applied
before paint and retained under `heterogeneous-batch-runtime.appearance.v1` when
storage is available. Blocked storage leaves a working in-memory choice. Appearance
changes preserve the selected data, timing boundary and linear chart scale.

Inter resolves from local installed faces. Visitors without Inter receive their
system font, with Arial as the final named fallback. The page makes no external
font request and does not distribute font files. A CSS family name alone does not
establish the actual glyph supplier. Browser checks can record resolved fonts
where the engine exposes that evidence. Forced colors, reduced motion, user text
scaling, labels, keyboard focus and the no-JavaScript tables remain part of review.

The chart gives every path its text label and value shown to five decimal places,
with full stored precision in the JSON and the same zero-based linear scale within
a selection. Missing CPU device-event intervals read
`Not measured`, not zero. Method identities stay stable when conditions change.
The table carries sample spread rather than presenting the median as a latency
guarantee. Memory tables retain process totals and shared-device free bytes as
different quantities.

`tests/browser/runtime-page.test.mjs` uses the existing Playwright route with an
owned loopback server, fresh contexts and headless-only launch. It performs no
native numerical work and blocks external page requests. Use the existing bounded
project verification controller for shared-PC execution. Source/static checks,
executed browser checks and a deployed URL journey are separate acceptance steps.
Native foreground and physical-display checks are outside that headless evidence.

The browser-only manifest and lock live in `tests/browser/` and pin Playwright
1.63.0. `RUNTIME_BROWSER_ENGINE` selects `chromium`, `webkit` or `firefox`.
The default is Chromium. `RUNTIME_BROWSER_CHANNEL=chrome` is an explicit local
Chromium-only adapter when the verification owner has reviewed that installation.
`RUNTIME_SCREENSHOT_DIR` optionally records private actual page captures. No
existing profile is attached and there is no headed fallback. The dependency
installation commands above are setup, not evidence that this machine installed
or executed all three engines.
