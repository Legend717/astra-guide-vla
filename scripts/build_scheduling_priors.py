"""Build approximate control-duration priors for ALL tasks and tag legacy experience without overwriting it.

The JSONL is a sampled language-training view, not a source of runtime oracle observations.
Transition oversampling is deduplicated. Short omitted segments and terminal padding remain limitations.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.codex_rgb_eval import write_json  # noqa: E402
from scripts.codex_supervisor.experience import applicability  # noqa: E402

PATTERN = re.compile(r"(?P<task>.+)_ep(?P<episode>\d+)_step(?P<step>\d+)\.png$")
LIMITATION = (
    "Approximate sampled simple_subgoal durations in actual controls; duplicates removed, demo excluded. "
    "Short segments omitted by the training sampler may be absorbed into adjacent labels; terminal padding "
    "can extend the final segment. Repeated identical labels are merged. Not a VLA safety limit or success criterion."
)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def read_samples(path):
    episodes = defaultdict(dict)
    order = []
    duplicates = 0
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            match = PATTERN.fullmatch(Path(row["images"][0]).name)
            if match is None:
                raise ValueError("unrecognized sample image path")
            task, episode, step = match["task"], int(match["episode"]), int(match["step"])
            if task not in order:
                order.append(task)
            label = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant")
            samples = episodes[task, episode]
            if step in samples:
                if samples[step] != label:
                    raise ValueError("conflicting labels at one task/episode/step")
                duplicates += 1
            samples[step] = label
    return episodes, order, duplicates


def duration_groups(episodes):
    groups = defaultdict(list)
    for (task, episode), mapping in episodes.items():
        ordered = sorted(mapping.items())
        first, label = ordered[0]
        for step, current in ordered[1:]:
            if current != label:
                groups[task, label].append(
                    {"episode": episode, "first": first, "last_exclusive": step, "steps": step - first}
                )
                first, label = step, current
        end = ordered[-1][0] + 1
        groups[task, label].append({"episode": episode, "first": first, "last_exclusive": end, "steps": end - first})
    return groups


def validate_pickle_samples(root, episodes, order, total_samples):
    """Verify source alignment on the first episode of each task using indexed binary search."""
    checks = []
    episode_offset = 0
    for task in order:
        local_episodes = sorted(ep for name, ep in episodes if name == task)
        episode = local_episodes[0]
        target = episode_offset + episode
        low, high = 0, total_samples
        while low < high:
            middle = (low + high) // 2
            with (root / "data" / f"{middle}.pkl").open("rb") as stream:
                item = pickle.load(stream)  # trusted local preprocessed dataset
            if int(item["epis_idx"][0]) < target:
                low = middle + 1
            else:
                high = middle
        with (root / "data" / f"{low}.pkl").open("rb") as stream:
            first = pickle.load(stream)
        start_step = int(first["step_idx"][0])
        samples = sorted(episodes[task, episode].items())
        for step, label in [samples[0], samples[len(samples) // 2], samples[-1]]:
            index = low + step - start_step
            with (root / "data" / f"{index}.pkl").open("rb") as stream:
                item = pickle.load(stream)
            valid = (
                int(item["epis_idx"][0]) == target
                and int(item["step_idx"][0]) == step
                and not bool(item["is_demo"][0])
                and item["simple_subgoal"] == label
            )
            checks.append({"task": task, "episode": episode, "step": step, "data_index": index, "matches": valid})
            if not valid:
                raise ValueError(f"raw metadata mismatch at {task}/{episode}/{step}")
        episode_offset += len(local_episodes)
    return checks


def import_experience(sources, annotations):
    libraries = defaultdict(list)
    mapping = []
    known = set()
    known_text = {}
    for source in sources:
        original = json.loads(source.read_text())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        scope = source.stem if source.stem not in ("general", "global") else "global"
        for index, entry in enumerate(original["entries"]):
            identifier = (
                entry.get("id")
                or "legacy-" + hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:24]
            )
            text_key = (entry.get("claim"), entry.get("conditions"), entry.get("source_run"))
            if identifier in known or text_key in known_text:
                mapping.append(
                    {
                        "source": str(source.resolve()),
                        "source_sha256": digest,
                        "source_index": index,
                        "id": identifier,
                        "duplicate_of": known_text.get(text_key, identifier),
                    }
                )
                continue
            known_text[text_key] = identifier
            known.add(identifier)
            tagged = {**entry, "id": identifier, "applicability": applicability(entry)}
            override = annotations.get(identifier)
            text_mentions_adviser = bool(
                re.search(
                    r"jev|advice|advisory|question|opinions|置信",
                    entry.get("claim", "") + " " + entry.get("conditions", ""),
                    re.IGNORECASE,
                )
            )
            if "supervision" in entry and not text_mentions_adviser:
                # Keep the factual text common; the original configuration stays in a separate JEV-only record.
                tagged = {k: v for k, v in tagged.items() if k != "supervision"}
                tagged["applicability"] = "common"
                libraries[scope].append(
                    {
                        **entry,
                        "id": identifier + ":legacy-supervision",
                        "applicability": "jev",
                        "derived_from": identifier,
                    }
                )
            libraries[scope].append(tagged)
            derived_id = None
            if override:
                derived_id = identifier + ":common"
                derived = {k: v for k, v in entry.items() if k not in ("supervision", "supersedes")}
                libraries[scope].append(
                    {**derived, **override, "id": derived_id, "applicability": "common", "derived_from": identifier}
                )
            mapping.append(
                {
                    "source": str(source.resolve()),
                    "source_sha256": digest,
                    "source_index": index,
                    "id": identifier,
                    "applicability": tagged["applicability"],
                    "scope": scope,
                    "common_derivative": derived_id,
                }
            )
    return libraries, mapping


def build(data_root, output, sources=(), annotations=None, *, validate_raw=True):
    if output.exists():
        raise ValueError("output directory must be new; never overwrite learned experience")
    source = data_root / "qwenvl/simple_subgoal_train.jsonl"
    episodes, order, duplicates = read_samples(source)
    groups = duration_groups(episodes)
    metadata = json.loads((data_root / "meta/stats.json").read_text())
    counted = sum(s["steps"] for segments in groups.values() for s in segments)
    if counted != metadata["execution_samples"]:
        raise ValueError("sample-derived execution total differs from dataset metadata")
    checks = validate_pickle_samples(data_root, episodes, order, counted) if validate_raw else []
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    stats = []
    libraries, migration = import_experience(sources, annotations or {})
    for (task, subgoal), segments in sorted(groups.items()):
        durations = [s["steps"] for s in segments]
        p25 = percentile(durations, 0.25)
        recommended = min(200, int(p25 // 10) * 10)
        close = recommended < 30
        summary = {
            "task": task,
            "subgoal": subgoal,
            "segments": len(durations),
            "episodes": len({s["episode"] for s in segments}),
            "min": min(durations),
            "p10": percentile(durations, 0.1),
            "p25": p25,
            "median": percentile(durations, 0.5),
            "p90": percentile(durations, 0.9),
            "max": max(durations),
            "recommended_steps": None if close else recommended,
            "close_review": close,
        }
        stats.append(summary)
        scheduling = {"subgoal": subgoal}
        if close:
            scheduling["close_review"] = True
        else:
            scheduling["recommended_steps"] = recommended
        identifier = "duration-" + hashlib.sha256(f"{fingerprint}:{task}:{subgoal}".encode()).hexdigest()[:24]
        libraries[task].append(
            {
                "id": identifier,
                "applicability": "step-budget",
                "kind": "hypothesis",
                "confidence": "low",
                "status": "active",
                "phase": subgoal,
                "claim": f"Demonstration control-duration prior: {json.dumps(summary, ensure_ascii=False)}",
                "conditions": LIMITATION,
                "scheduling": scheduling,
                "provenance": {
                    "kind": "demonstration_prior",
                    "task": task,
                },
            }
        )
    output.mkdir(parents=True)
    for task in ["global", *order]:
        target = output / "experience" / ("general.json" if task == "global" else f"tasks/{task}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, {"schema_version": 3, "version": 1, "entries": libraries[task], "commits": []})
    report = {
        "source": str(source),
        "source_sha256": fingerprint,
        "task_count": len(order),
        "episode_count": len(episodes),
        "execution_samples": counted,
        "duplicate_rows_removed": duplicates,
        "limitations": LIMITATION,
        "raw_validation": checks,
        "statistics": stats,
    }
    write_json(output / "duration-statistics.json", report)
    write_json(output / "experience-migration.json", {"mapping": migration})
    with (output / "duration-statistics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    print(
        json.dumps(
            {
                "output": str(output),
                "tasks": len(order),
                "episodes": len(episodes),
                "groups": len(stats),
                "execution_samples": counted,
                "raw_checks": len(checks),
            },
            ensure_ascii=False,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experience-source", type=Path, action="append", default=[])
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--skip-raw-validation", action="store_true")
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text()) if args.annotations else {}
    build(
        args.data_root, args.output_dir, args.experience_source, annotations, validate_raw=not args.skip_raw_validation
    )


if __name__ == "__main__":
    main()
