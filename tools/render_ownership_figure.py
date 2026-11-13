"""Render a conceptual ownership explanation from the retained public contract.

Generation requires --font-dir or HBR_FONT_DIR with genuine static Inter faces.
--check reads the committed inputs and exports without loading plotting libraries.
No workload, native package, model or device is executed by this renderer.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = "docs/runtime-ownership-input.json"
RECORD = "docs/runtime-ownership-figure.json"
PALETTE = "docs/figure-tokens-obscur.json"
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

# Obscur figure rules. Type sizes are points at 100 dpi. Spacing is in pixels.
PT = 72 / 100
EDITIONS = {
    False: {"name": "runtime-ownership-obscur", "width": 1800, "margin": 72,
            "sizes_pt": {"title": 40, "subtitle": 30, "card_title": 32, "body": 30, "footer": 30},
            "card_width": 500, "card_gap": 78},
    True: {"name": "runtime-ownership-obscur-narrow", "width": 960, "margin": 48,
           "sizes_pt": {"title": 40, "subtitle": 33, "card_title": 34, "body": 33, "footer": 33},
           "card_width": 864, "card_gap": 24},
}
SPACING = {"card_side": 28, "card_top": 32, "card_bottom": 28, "title_to_body": 20,
           "title_to_subtitle": 16, "before_cards": 48, "before_footer": 48,
           "footer_lines": 14, "below_footer": 49, "radius": 18, "outline": 2,
           "connector": 2.5, "line_spacing": 1.3}
TITLE = "Keep the work alive."
SUBTITLE = "A masked sum with owned input and result."
CARDS = {
    False: (("1  Capture input", "Values [2, -3, 5]\nMask [1, 0, 1]"),
            ("2  Own the work", "Selected: 2 + 5\nOwned storage."),
            ("3  Return result", "7.0\nAfter return.")),
    True: (("1  Capture input", "Values [2, -3, 5]\nMask [1, 0, 1]"),
           ("2  Own the work", "Selected entries: 2 + 5"),
           ("3  Return result", "7.0 after the computation returns")),
}
# The rule line stays in text and the note in muted, so weight and tone set the
# cancellation rule apart without a hue.
FOOTER = {
    False: ("Cancellation is a request. Storage stays owned until physical return.",
            "Conceptual example. The uncancelled sum is 7.0, not a timing or CPU/GPU claim."),
    True: ("Cancellation is a request.\nStorage stays owned until physical return.",
           "Conceptual example, not timing.\nThe uncancelled sum is 7.0."),
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


def palette(root=ROOT):
    data = json.loads((root / PALETTE).read_text(encoding="utf-8"))
    for role, value in data["Obscur"].items():
        if len(value) != 7 or not value[1:3] == value[3:5] == value[5:7]:
            raise ValueError("Obscur carries no hue: " + role)
    return data


def check_outputs(root=ROOT):
    data = inputs(root)
    record = json.loads((root / RECORD).read_text(encoding="utf-8"))
    if record["evidence"] != data or record["new_measurement"] is not False:
        raise ValueError("Ownership figure provenance changed")
    if record["palette"] != palette(root)["source"]:
        raise ValueError("Ownership figure palette source changed")
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

    result, evidence, shared = {}, {}, set()
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
            em = font["head"].unitsPerEm
            shared.add((font["hhea"].ascent / em, -font["hhea"].descent / em))
        result[label] = lambda points, p=path: FontProperties(fname=str(p), size=points)
        evidence[label] = {"postscript": postscript, "weight": weight,
                           "sha256": digest(path), "italic": False}
    if len(shared) != 1:
        raise ValueError("The three faces must share one set of vertical metrics")
    ascent, descent = shared.pop()
    return result, evidence, {"ascent_em": ascent, "descent_em": descent}


def draw(data, roles, faces, metrics, *, narrow, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    edition, space = EDITIONS[narrow], SPACING
    sizes, width, margin = edition["sizes_pt"], edition["width"], edition["margin"]
    pitch = space["line_spacing"]
    # The canvas starts tall and shrinks to the measured footer. One data unit
    # stays one pixel and every block hangs from its top, so nothing moves.
    height = 2000
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                         facecolor=roles["canvas"])
    axes = figure.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(height, 0))
    axes.axis("off")
    renderer = figure.canvas.get_renderer()
    scratch = axes.text(0, 0, "")

    def measure(value, size, face):
        scratch.set_text(value)
        scratch.set_fontproperties(faces[face](size))
        return scratch.get_window_extent(renderer).width

    def greedy(paragraph, size, face, room):
        lines, line = [], ""
        for word in paragraph.split():
            if measure(word, size, face) > room:
                raise ValueError("An ownership label word exceeds its room: " + word)
            candidate = (line + " " + word).strip()
            if measure(candidate, size, face) <= room:
                line = candidate
            else:
                lines.append(line)
                line = word
        return lines + [line]

    def balanced(paragraph, size, face, room):
        # Keep the greedy line count, then narrow the measure until one more
        # pixel would add a line. This avoids a lone last word.
        lines = greedy(paragraph, size, face, room)
        if len(lines) == 1:
            return lines
        low, high = 1, int(room)
        while low < high:
            middle = (low + high) // 2
            try:
                fits = len(greedy(paragraph, size, face, middle)) == len(lines)
            except ValueError:
                fits = False
            low, high = (low, middle) if fits else (middle + 1, high)
        return greedy(paragraph, size, face, low)

    def place(x, top, value, size, face, role, room, *, wrap=True):
        lines = []
        for paragraph in value.split("\n"):
            if wrap:
                lines += balanced(paragraph, size, face, room)
            elif measure(paragraph, size, face) > room:
                raise ValueError("An ownership label exceeds its room: " + paragraph)
            else:
                lines.append(paragraph)
        # Baselines come from the font metrics, not from the ink of each line,
        # so labels with and without ascenders or descenders still align.
        em = size / PT
        first = top + metrics["ascent_em"] * em
        artists = [axes.text(x, first + index * pitch * em, line, va="baseline",
                             fontproperties=faces[face](size), color=roles[role])
                   for index, line in enumerate(lines)]
        bottom = first + (len(lines) - 1) * pitch * em + metrics["descent_em"] * em
        return {"lines": artists, "top": top, "bottom": bottom}

    def ink(block):
        inverse = axes.transData.inverted()
        boxes = []
        for label in block["lines"]:
            extent = label.get_window_extent(renderer)
            left, bottom = inverse.transform((extent.x0, extent.y0))
            right, top = inverse.transform((extent.x1, extent.y1))
            boxes.append((left, top, right, bottom))
        return tuple(float(pick(values)) for pick, values in
                     zip((min, min, max, max), zip(*boxes)))

    room = width - 2 * margin
    title = place(margin, margin, TITLE, sizes["title"], "Bold", "text", room)
    subtitle = place(margin, title["bottom"] + space["title_to_subtitle"], SUBTITLE,
                     sizes["subtitle"], "Regular", "muted", room)
    cards_top = math.ceil(subtitle["bottom"] + space["before_cards"])
    card_width, gap = edition["card_width"], edition["card_gap"]
    side, top_pad = space["card_side"], space["card_top"]
    cards, top = [], cards_top
    for index, (heading_text, body_text) in enumerate(CARDS[narrow]):
        x = margin if narrow else margin + index * (card_width + gap)
        y = top if narrow else cards_top
        heading = place(x + side, y + top_pad, heading_text, sizes["card_title"], "SemiBold",
                        "text", card_width - 2 * side, wrap=False)
        body = place(x + side, heading["bottom"] + space["title_to_body"], body_text,
                     sizes["body"], "Regular", "text", card_width - 2 * side)
        card_height = math.ceil(body["bottom"] + space["card_bottom"] - y)
        cards.append({"x": x, "y": y, "height": card_height, "heading": heading, "body": body})
        top = y + card_height + gap
    if not narrow:
        # Wide cards share the tallest measured height so they read as one row.
        common = max(card["height"] for card in cards)
        for card in cards:
            card["height"] = common
    cards_bottom = max(card["y"] + card["height"] for card in cards)
    rule = place(margin, cards_bottom + space["before_footer"], FOOTER[narrow][0],
                 sizes["footer"], "SemiBold", "text", room)
    note = place(margin, rule["bottom"] + space["footer_lines"], FOOTER[narrow][1],
                 sizes["footer"], "SemiBold", "muted", room)
    scratch.remove()
    height = math.ceil(note["bottom"] + space["below_footer"])
    figure.set_size_inches(width / 100, height / 100)
    axes.set_ylim(height, 0)

    for card in cards:
        axes.add_patch(FancyBboxPatch(
            (card["x"], card["y"]), card_width, card["height"],
            boxstyle="round,pad=0,rounding_size=" + str(space["radius"]),
            linewidth=space["outline"] * PT, edgecolor=roles["border"],
            facecolor=roles["control"], zorder=0))
    for before, after in zip(cards, cards[1:]):
        if narrow:
            middle = margin + card_width / 2
            start = (middle, before["y"] + before["height"] + 2)
            end = (middle, after["y"] - 2)
            head = 14
        else:
            middle = cards_top + cards[0]["height"] / 2
            start = (before["x"] + card_width + 10, middle)
            end = (after["x"] - 10, middle)
            head = 20
        axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head,
                                       shrinkA=0, shrinkB=0, color=roles["text"],
                                       linewidth=space["connector"] * PT))
    metadata = {"Description": json.dumps({"evidence": data, "new_measurement": False},
                                          sort_keys=True)}
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for block in (title, subtitle, rule, note, *(card[kind] for card in cards
                                                  for kind in ("heading", "body"))):
        left, upper, right, lower = ink(block)
        if left < 0 or upper < 0 or right > width or lower > height:
            raise ValueError("A diagram label escaped its canvas: " +
                             " ".join(label.get_text() for label in block["lines"]))
    geometry = []
    for card in cards:
        inner = (card["x"] + side, card["y"] + top_pad, card["x"] + card_width - side,
                 card["y"] + card["height"] - space["card_bottom"])
        measured = {}
        for kind in ("heading", "body"):
            block = card[kind]
            extent = ink(block)
            measured[kind] = [round(value, 3) for value in extent]
            upper = min(extent[1], block["top"])
            lower = max(extent[3], block["bottom"])
            if (extent[0] < inner[0] - .01 or upper < inner[1] - .01 or
                    extent[2] > inner[2] + .01 or lower > inner[3] + .01):
                raise ValueError("A diagram label escaped its padded panel: " +
                                 block["lines"][0].get_text() + " measured=" + str(extent) +
                                 " permitted=" + str(inner))
        if card["body"]["top"] < card["heading"]["bottom"] + space["title_to_body"] - .01:
            raise ValueError("A diagram body overlaps its heading: " +
                             card["heading"]["lines"][0].get_text())
        geometry.append({"card": [card["x"], card["y"], card_width, card["height"]],
                         "padded_panel": list(inner), "measured_labels": measured})
    figure.savefig(output.with_suffix(".png"), facecolor=roles["canvas"],
                   metadata={"Description": metadata["Description"]})
    svg_path = output.with_suffix(".svg")
    with svg_path.open("wb") as stream:
        figure.savefig(stream, format="svg", facecolor=roles["canvas"],
                       metadata={**metadata, "Date": None})
    svg_path.write_bytes(svg_path.read_bytes().replace(b"\r\n", b"\n"))
    plt.close(figure)
    return {"width": width, "height": height, "margin": margin,
            "sizes_pt": sizes, "panel_geometry": geometry}


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
    faces, font_record, metrics = fonts(Path(args.font_dir))
    tokens = palette()
    layouts = {}
    for narrow in (False, True):
        name = EDITIONS[narrow]["name"]
        layouts[name] = draw(data, tokens["Obscur"], faces, metrics, narrow=narrow,
                             output=ROOT / "docs" / name)
    record = {"schema_version": 2, "evidence": data, "new_measurement": False,
              "palette": tokens["source"],
              "palette_roles": ["canvas", "text", "muted", "border", "control"],
              "renderer": "tools/render_ownership_figure.py", "spacing_px": SPACING,
              "layouts": layouts,
              "typography": {"files": font_record, "vertical_metrics": metrics,
                             "svg": "Outlined labels", "font_binaries_distributed": False},
              "inputs_sha256_lf": {name: digest(ROOT / name, text=True) for name in INPUTS},
              "outputs_sha256": {name: digest(ROOT / name) for name in FILES}}
    (ROOT / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Rendered conceptual ownership editions; no numerical or native work executed")


if __name__ == "__main__":
    main()
