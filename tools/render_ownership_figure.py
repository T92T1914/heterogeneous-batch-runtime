"""Render a conceptual ownership explanation from the retained public contract.

Generation requires --font-dir or HBR_FONT_DIR with genuine static Inter faces.
--check reads the committed inputs and exports without loading plotting libraries.
No workload, native package, model or device is executed by this renderer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = "docs/runtime-ownership-input.json"
RECORD = "docs/runtime-ownership-figure.json"
PALETTE = "web/theme-tokens.json"
INPUTS = (INPUT, PALETTE, "tools/render_ownership_figure.py")
FILES = tuple("docs/runtime-ownership-obscur" + suffix for suffix in
              (".png", ".svg", "-narrow.png", "-narrow.svg"))
FACES = {"Regular": ("Inter-Regular.ttf", "Inter-Regular", 400),
         "SemiBold": ("Inter-SemiBold.ttf", "Inter-SemiBold", 600),
         "Bold": ("Inter-Bold.ttf", "Inter-Bold", 700)}
FONT_HASHES = {
    "Regular": "40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82",
    "SemiBold": "78a843fade9d4612a5567302fb595b56976eb5fcebf4fea5a5912d638bafcde3",
    "Bold": "288316099b1e0a47a4716d159098005eef7c0066921f34e3200393dbdb01947f",
}


def digest(path, *, text=False):
    value = path.read_text(encoding="utf-8").encode() if text else path.read_bytes()
    return hashlib.sha256(value).hexdigest()


def inputs(root=ROOT):
    data = json.loads((root / INPUT).read_text(encoding="utf-8"))
    if data["example"] != {"values": [2, -3, 5], "mask": [1, 0, 1],
                           "selected": [2, 5], "sum": 7}:
        raise ValueError("Conceptual arithmetic must agree with the retained example")
    if data["kind"] != "conceptual mechanism and worked arithmetic, not measured execution":
        raise ValueError("A conceptual figure must not become an execution record")
    return data


def check_outputs(root=ROOT):
    data = inputs(root)
    record = json.loads((root / RECORD).read_text(encoding="utf-8"))
    if record["evidence"] != data or record["new_measurement"] is not False:
        raise ValueError("Ownership figure provenance changed")
    for label, expected in FONT_HASHES.items():
        if record["typography"]["files"][label]["sha256"] != expected:
            raise ValueError("Ownership font identity changed: " + label)
    for name in INPUTS:
        if record["inputs_sha256_lf"][name] != digest(root / name, text=True):
            raise ValueError("Ownership figure input changed: " + name)
    if set(record["outputs_sha256"]) != set(FILES):
        raise ValueError("Wide and narrow PNG/SVG editions are required")
    for name, expected in record["outputs_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Ownership figure output changed: " + name)
    return record


def fonts(directory):
    from fontTools.ttLib import TTFont
    from matplotlib.font_manager import FontProperties

    result, evidence = {}, {}
    for label, (filename, postscript, weight) in FACES.items():
        path = directory / filename
        if digest(path) != FONT_HASHES[label]:
            raise ValueError("Expected the pinned official Inter 4.1 face: " + filename)
        with TTFont(path) as font:
            names = {entry.toUnicode() for entry in font["name"].names if entry.nameID == 6}
            if postscript not in names or font["OS/2"].usWeightClass != weight:
                raise ValueError("Expected the genuine static face: " + filename)
            if font["OS/2"].fsSelection & 1:
                raise ValueError("An upright face was required: " + filename)
            # The diagram uses only these basic public labels and arithmetic.
            if not set(range(32, 127)).issubset(font.getBestCmap()):
                raise ValueError("Missing Latin label coverage: " + filename)
        result[label] = lambda size, p=path: FontProperties(fname=str(p), size=size * .72)
        evidence[label] = {"postscript": postscript, "weight": weight,
                           "sha256": digest(path), "italic": False}
    return result, evidence


def draw(data, roles, faces, *, narrow, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    width, height = (960, 1280) if narrow else (1800, 800)
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                         facecolor=roles["canvas"])
    axes = figure.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(height, 0))
    axes.axis("off")
    panels = []

    def text(x, y, value, size=36, face="Regular", role="text"):
        return axes.text(x, y, value, fontproperties=faces[face](size),
                         color=roles[role], va="top", linespacing=1.35)

    def box(x, y, w, h, title, body):
        axes.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16",
                                      linewidth=2, edgecolor=roles["border"],
                                      facecolor=roles["panel"]))
        heading = text(x + 28, y + 24, title, 40 if narrow else 46, "SemiBold", "accent")
        label = text(x + 28, y + 96, body, 40 if narrow else 46)
        panels.append({"heading": heading, "body": label,
                       "inner": (x + 28, y + 24, x + w - 28, y + h - 20),
                       "body_start": y + 96})

    text(60, 35, "HETEROGENEOUS BATCH RUNTIME", 30, "SemiBold", "accent")
    text(60, 90, "Keep the work alive.", 62, "Bold")
    text(60, 205, "A masked sum with owned input and result.",
         36 if narrow else 38)
    if narrow:
        box(60, 285, 840, 200, "1  Capture input", "Values [2, -3, 5]   Mask [1, 0, 1]")
        box(60, 525, 840, 200, "2  Own the work", "Selected entries: 2 + 5")
        box(60, 765, 840, 200, "3  Return result", "7.0 after the computation returns")
        for y in (490, 730):
            axes.add_patch(FancyArrowPatch((480, y), (480, y + 25),
                                           arrowstyle="-|>", mutation_scale=22,
                                           color=roles["muted"], linewidth=2))
        text(60, 1005, "Cancellation is a request.", 40, "SemiBold", "warning")
        text(60, 1075, "Storage stays owned until physical return.", 38)
        text(60, 1150, "Conceptual example, not timing.\nThe uncancelled sum is 7.0.", 32, role="muted")
    else:
        box(60, 280, 520, 300, "1  Capture input", "Values [2, -3, 5]\nMask [1, 0, 1]")
        box(640, 280, 520, 300, "2  Own the work", "Selected: 2 + 5\nOwned storage.")
        box(1220, 280, 520, 300, "3  Return result", "7.0\nAfter return.")
        for x in (590, 1170):
            axes.add_patch(FancyArrowPatch((x, 430), (x + 40, 430), arrowstyle="-|>",
                                           mutation_scale=24, color=roles["muted"], linewidth=2))
        text(60, 620, "Cancellation is a request. Storage stays owned until physical return.",
             38, "SemiBold", "warning")
        text(60, 700, "Conceptual example. The uncancelled sum is 7.0, not a timing or CPU/GPU claim.",
             32, role="muted")
    metadata = {"Description": json.dumps({"evidence": data, "new_measurement": False},
                                          sort_keys=True)}
    figure.canvas.draw()

    def data_bounds(label):
        extent = label.get_window_extent(figure.canvas.get_renderer())
        left, bottom = axes.transData.inverted().transform((extent.x0, extent.y0))
        right, top = axes.transData.inverted().transform((extent.x1, extent.y1))
        return tuple(float(value) for value in (left, top, right, bottom))

    def wrap_body(panel):
        label = panel["body"]
        max_width = panel["inner"][2] - panel["inner"][0]
        renderer = figure.canvas.get_renderer()
        lines = []
        for paragraph in label.get_text().split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                label.set_text(candidate)
                if label.get_window_extent(renderer).width <= max_width:
                    line = candidate
                elif line:
                    lines.append(line)
                    line = word
                    label.set_text(word)
                    if label.get_window_extent(renderer).width > max_width:
                        raise ValueError("An ownership label word exceeds its panel: " + word)
                else:
                    raise ValueError("An ownership label word exceeds its panel: " + word)
            lines.append(line)
        label.set_text("\n".join(lines))

    # Inter's actual line box can be taller than its configured pixel size.
    # Place the body from the measured heading bottom, then verify the real
    # padded panel geometry instead of guessing a separate heading height.
    for panel in panels:
        wrap_body(panel)
        heading_bottom = data_bounds(panel["heading"])[3]
        x, _ = panel["body"].get_position()
        panel["body"].set_position((x, max(panel["body_start"], heading_bottom + 16)))
    figure.canvas.draw()
    bounds = figure.bbox
    for label in axes.texts:
        extent = label.get_window_extent(figure.canvas.get_renderer())
        if extent.x0 < 0 or extent.x1 > bounds.width or extent.y0 < 0 or extent.y1 > bounds.height:
            raise ValueError("A diagram label escaped its canvas: " + label.get_text())
    geometry = []
    for panel in panels:
        inner = panel["inner"]
        measured = {}
        for kind in ("heading", "body"):
            label = panel[kind]
            extent = data_bounds(label)
            measured[kind] = [round(value, 3) for value in extent]
            if (extent[0] < inner[0] - .01 or extent[1] < inner[1] - .01 or
                    extent[2] > inner[2] + .01 or extent[3] > inner[3] + .01):
                raise ValueError("A diagram label escaped its padded panel: " +
                                 label.get_text() + " measured=" + str(extent) +
                                 " permitted=" + str(inner))
        if measured["body"][1] < measured["heading"][3] + 16 - .01:
            raise ValueError("A diagram body overlaps its heading: " +
                             panel["heading"].get_text() + " measured=" + str(measured))
        geometry.append({"padded_panel": inner, "measured_labels": measured})
    figure.savefig(output.with_suffix(".png"), facecolor=roles["canvas"],
                   metadata={"Description": metadata["Description"]})
    svg_path = output.with_suffix(".svg")
    with svg_path.open("wb") as stream:
        figure.savefig(stream, format="svg", facecolor=roles["canvas"],
                       metadata={**metadata, "Date": None})
    svg_path.write_bytes(svg_path.read_bytes().replace(b"\r\n", b"\n"))
    plt.close(figure)
    return {"width": width, "height": height, "panel_geometry": geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, default=os.environ.get("HBR_FONT_DIR"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_outputs()
        print("Ownership inputs and four committed exports match; no workload executed")
        return
    if args.font_dir is None:
        parser.error("Supply --font-dir or HBR_FONT_DIR; fonts are never bundled")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "hbr-ownership-v1"})
    data = inputs()
    faces, font_record = fonts(Path(args.font_dir))
    roles = json.loads((ROOT / PALETTE).read_text(encoding="utf-8"))["themes"]["Obscur"]
    layouts = {}
    for narrow in (False, True):
        name = "runtime-ownership-obscur" + ("-narrow" if narrow else "")
        layouts[name] = draw(data, roles, faces, narrow=narrow, output=ROOT / "docs" / name)
    record = {"schema_version": 1, "evidence": data, "new_measurement": False,
              "theme_revision": "7a57fe750ff50205a17e1d342106a0d3f2777159",
              "renderer": "tools/render_ownership_figure.py", "layouts": layouts,
              "typography": {"files": font_record, "svg": "Outlined labels", "font_binaries_distributed": False},
              "inputs_sha256_lf": {name: digest(ROOT / name, text=True) for name in INPUTS},
              "outputs_sha256": {name: digest(ROOT / name) for name in FILES}}
    (ROOT / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Rendered conceptual ownership editions; no numerical or native work executed")


if __name__ == "__main__":
    main()
