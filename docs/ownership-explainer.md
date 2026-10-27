# Own the work until it returns

For values `[2, -3, 5]` and mask `[1, 0, 1]`, the selected entries are `2` and `5`.
Their ordinary uncancelled sum is `7.0`. The README and public reader show this
small arithmetic example beside the runtime's separate ownership rule.

A running cancellation request records intent. It does not interrupt an arbitrary
callable, confirm physical return or permit early reuse of its captured storage.
Queued work can be cancelled before it starts. The synchronous native API borrows
immutable spans through return, while the Python boundary snapshots admitted
inputs before releasing the GIL. Those API boundaries remain in the
[contracts](contracts.md), [runtime](../src/runtime.cpp) and
[Python binding](../src/bindings.cpp).

The diagram is conceptual. It is not a new native run, a timing observation or a
CPU/GPU comparison, and it does not promise that a cancelled request publishes
the illustrated result. The numerical and performance records retain their own
source and hardware conditions.

The Obscur [wide](runtime-ownership-obscur.png) and
[narrow](runtime-ownership-obscur-narrow.png) renditions carry the same example.
[Editable generator](../tools/render_ownership_figure.py),
[claim inputs](runtime-ownership-input.json) and
[rendering record](runtime-ownership-figure.json) retain the source and font
identities. SVG editions preserve labels as outlines. No fonts are distributed.

To regenerate in an isolated environment with the plotting dependencies and
genuine static Inter Regular, SemiBold and Bold files:

```sh
python -m pip install -r requirements-figures.txt
python tools/render_ownership_figure.py --font-dir "$INTER_FONT_DIR"
python tools/render_ownership_figure.py --check
python tools/build_site.py --write
```

The public page opens in Obscur and remains Obscur without JavaScript. Explicit
Clair and Auto choices remain available. Auto follows the system appearance and
is now saved explicitly. Older versions stored Auto by removing the preference,
so an old Auto selection and a first visit cannot be distinguished. Both receive
the new default until the visitor selects a preference. Saved Clair and Obscur
choices remain effective; blocked storage still permits in-page switching.
