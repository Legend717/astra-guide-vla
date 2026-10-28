"""Explicit GPT-selected image coordinates; no perception or task solver."""

import json
import math
from pathlib import Path
import re

TEMPLATES = json.loads(Path(__file__).with_name("templates.json").read_text())


def matches_template(text: str, template: str) -> bool:
    pattern = re.escape(template)
    for field in re.findall(r"\[([^]]+)\]", template):
        choices = field.split("/")
        replacement = "(?:" + "|".join(re.escape(x) for x in choices) + ")"
        if len(choices) == 1 or "..." in choices:
            replacement = r"[A-Za-z0-9 -]+"
        pattern = pattern.replace(re.escape(f"[{field}]"), replacement)
    return re.fullmatch(pattern, text) is not None


def convert(task: str, specification: dict, image_hw: tuple[int, int]) -> dict:
    """Use ordered <point> placeholders, never ambiguous inline numeric coordinates.

    The legacy predictor scales by H/W (not H-1/W-1). Clamp the exact 1000
    endpoint to the last valid pixel rather than producing an off-image index.
    """
    if not isinstance(specification, dict):
        raise ValueError("subgoal must be an object with text and point_yx_1000")
    if set(specification) - {"text", "point_yx_1000", "simple_subgoal"}:
        raise ValueError("subgoal contains unsupported fields")
    text = specification.get("text")
    points = specification.get("point_yx_1000", [])
    if not isinstance(text, str) or not isinstance(points, list):
        raise ValueError("text and point_yx_1000 list are required")
    if text.count("<point>") != len(points):
        raise ValueError("one ordered point_yx_1000 pair per <point> is required")
    normalized = text.replace("<point>", "<y, x>")
    if task not in TEMPLATES or not any(matches_template(normalized, t) for t in TEMPLATES[task]["grounded"]):
        raise ValueError("text does not match a supported task template")
    if len(image_hw) != 2 or any(type(n) is not int or n <= 0 for n in image_hw):
        raise ValueError("original front image height and width must be positive integers")
    pixels = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("each point_yx_1000 must be [y, x]")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in point):
            raise ValueError("point coordinates must be finite numbers in [0, 1000]")
        pixel = [min(int(v * size / 1000), size - 1) for v, size in zip(point, image_hw, strict=True)]
        pixels.append(pixel)
        text = text.replace("<point>", f"<{pixel[0]}, {pixel[1]}>", 1)
    simple = specification.get("simple_subgoal")
    if simple is not None and (
        not isinstance(simple, str) or not any(matches_template(simple, t) for t in TEMPLATES[task]["simple"])
    ):
        raise ValueError("simple_subgoal does not match a supported simple template")
    return {"grounded_subgoal": text, "simple_subgoal": simple, "pixel_yx": pixels, "image_hw": list(image_hw)}


def annotate(image_path: Path, output_path: Path, pixels: list[list[int]]) -> None:
    """Overlay only supplied GPT points on an already observed image."""
    from PIL import Image
    from PIL import ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, (y, x) in enumerate(pixels):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="red", width=2)
        draw.text((x + 5, y), str(index), fill="red")
    image.save(output_path)
