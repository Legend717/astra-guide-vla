"""Video labels describe executed actions only; source RGB remains untouched."""

import json
import textwrap

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from scripts import codex_rgb_eval as rgb


def labeled_frame(front, wrist, frame_id, context, task):
    canvas = Image.new("RGB", (1024, 768), (16, 20, 28))
    draw = ImageDraw.Draw(canvas)
    source = context["source"]
    color = {"VLA": "#2474b5", "INTERVENTION": "#b65016", "RESET": "#535b64"}[source]
    draw.rectangle((0, 0, 1024, 36), fill=color)
    title = f"{source} | frame {frame_id} | request {context.get('request_id', 'reset')}"
    lines = f"Task: {task}\nSubgoal: {context.get('subgoal', '(not set)')}"
    for size in range(20, 7, -1):
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size)
        except OSError:
            font = ImageFont.load_default(size=size)
        columns = max(1, int(1000 / font.getlength("M")))
        wrapped = [line for paragraph in lines.splitlines() for line in textwrap.wrap(paragraph, columns)]
        if len(wrapped) * (size + 4) <= 196:
            break
    draw.text((12, 8), title, font=font, fill="white")
    draw.multiline_text((12, 44), "\n".join(wrapped), font=font, fill="white", spacing=4)
    draw.text((12, 234), "FRONT", fill="white")
    draw.text((524, 234), "WRIST", fill="white")
    for index, original in enumerate((front, wrist)):
        panel = original.convert("RGB")
        panel.thumbnail((512, 512))
        # Enlarge small benchmark images only in the viewing video, not policy/RGB archives.
        scale = min(512 / panel.width, 512 / panel.height)
        panel = panel.resize((round(panel.width * scale), round(panel.height * scale)))
        canvas.paste(panel, (index * 512, 256))
    return canvas


def render_archive(directory):
    """Rebuild a closed run's viewing video from actual archived frames and control events."""
    import imageio.v2 as imageio
    import numpy as np

    result = rgb.read_json(directory / "result.json")
    if result["lifecycle"] != "closed":
        raise ValueError("video can only be rebuilt after the run closes")
    manifest = rgb.read_json(directory / "manifest.json")
    task = rgb.read_json(directory / "latest.json")["task_goal"]
    contexts = {}
    subgoal = "(not set)"
    for line in (directory / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "prompt":
            subgoal = event["converted"]["grounded_subgoal"]
        elif event["event"] == "control_return":
            for frame_id in range(event["before_frame"] + 1, event["after_frame"] + 1):
                contexts[frame_id] = {
                    "source": "VLA" if event["kind"] == "joint" else "INTERVENTION",
                    "request_id": event["request_id"],
                    "subgoal": subgoal,
                }
    folder = directory / "video"
    folder.mkdir(exist_ok=True)
    temporary = folder / "rollout-labeled.tmp.mp4"
    count = 0
    with imageio.get_writer(temporary, fps=manifest["fps"], codec="libx264", quality=8, macro_block_size=1) as writer:
        for line in (directory / "frames.jsonl").read_text().splitlines():
            frame = json.loads(line)
            context = frame.get("execution", contexts.get(frame["frame_id"]))
            if context is None:
                if frame["batch"] != "reset":
                    raise ValueError("non-reset frame has no executed action provenance")
                context = {"source": "RESET", "request_id": "reset", "subgoal": "(not set)"}
            with (
                Image.open(directory / frame["rgb"]["front"]) as front,
                Image.open(directory / frame["rgb"]["wrist"]) as wrist,
            ):
                writer.append_data(np.asarray(labeled_frame(front, wrist, frame["frame_id"], context, task)))
            count += 1
    if count != result["recorded_frames"]:
        raise ValueError("video source frame count mismatch")
    target = directory / "rollout.mp4"
    backup = folder / "rollout-original.mp4"
    if target.exists() and not backup.exists():
        import shutil

        shutil.copy2(target, backup)
    temporary.replace(target)
    return {"video": str(target), "frames": count, "labels": ["source", "request_id", "task", "subgoal"]}
