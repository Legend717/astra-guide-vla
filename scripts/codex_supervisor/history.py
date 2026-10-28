"""Render complete public RGB history for one decision, without advancing simulation."""

import json
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

from scripts import codex_rgb_eval as rgb


def render_history(directory: Path, first: int, last: int, *, selected=None) -> dict:
    if first < 0 or last < first:
        raise ValueError("invalid inclusive frame range")
    frames = [
        frame
        for line in (directory / "frames.jsonl").read_text().splitlines()
        if first <= (frame := json.loads(line))["frame_id"] <= last
    ]
    if [frame["frame_id"] for frame in frames] != list(range(first, last + 1)):
        raise ValueError("requested history contains missing frames")
    suffix = ""
    if selected is not None:
        frames = [frame for frame in frames if frame["frame_id"] in selected]
        suffix = "-sample-" + "-".join(str(index) for index in selected)
    folder = directory / "history" / f"frames-{first:07d}-{last:07d}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    sheets = []
    for offset in range(0, len(frames), 8):
        page = frames[offset : offset + 8]
        sheet = Image.new("RGB", (1024, 280 * ((len(page) + 1) // 2)))
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(page):
            x, y = index % 2 * 512, index // 2 * 280
            draw.text((x + 4, y + 4), f"frame {frame['frame_id']} | front / wrist", fill="white")
            for camera_index, camera in enumerate(("front", "wrist")):
                with Image.open(directory / frame["rgb"][camera]) as source:
                    tile = source.convert("RGB")
                    tile.thumbnail((256, 256))
                    sheet.paste(tile, (x + camera_index * 256, y + 24))
        path = folder / f"sheet-{offset // 8:04d}.jpg"
        sheet.save(path, quality=95)
        sheets.append(str(path.relative_to(directory)))
    result = {
        "first_frame": first,
        "last_frame": last,
        "frame_count": len(frames),
        "contact_sheets": sheets,
        "frames": [{"frame_id": frame["frame_id"], "rgb": frame["rgb"]} for frame in frames],
        "note": "Chronological front/wrist pairs. Original RGB paths retained; no inference or env.step. "
        + (
            "Only selected frames shown; omitted frames are NOT reviewed."
            if selected is not None
            else "All frames shown."
        ),
    }
    rgb.write_json(folder / "index.json", result)
    return result


def render_sampled_history(directory: Path, first: int, last: int, max_frames: int = 8) -> dict:
    if (
        any(type(v) is not int for v in (first, last, max_frames))
        or not 0 <= first <= last
        or not 2 <= max_frames <= 32
    ):
        raise ValueError("sample requires a valid inclusive range and max_frames in 2..32")
    count = min(max_frames, last - first + 1)
    selected = [first] if count == 1 else [first + (last - first) * i // (count - 1) for i in range(count)]
    result = render_history(directory, first, last, selected=selected)
    return {**result, "selected_frames": selected, "omitted_frame_count": last - first + 1 - count}
