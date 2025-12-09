"""Convert MALT + ARB trajectories to training format."""

import json
import random
from pathlib import Path

MALT_HACK_LABELS = {
    "bypass_constraints",
    "hardcoded_solution",
    "ignores_task_instructions",
    "sabotage",
}


def extract_text_from_malt(data):
    text_parts = []
    # MALT uses 'nodes' with node_data.message.content structure
    nodes = data.get("nodes", [])
    for node in nodes:
        if isinstance(node, dict):
            node_data = node.get("node_data", {})
            if isinstance(node_data, dict):
                msg = node_data.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        text_parts.append(content)
    # Also check transcript/messages as fallback
    transcript = data.get("transcript", data.get("messages", []))
    if isinstance(transcript, str):
        try:
            transcript = json.loads(transcript)
        except:
            transcript = [{"content": transcript}]
    for msg in transcript if isinstance(transcript, list) else []:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
    return "\n".join(text_parts)


def convert_malt(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
    except:
        return None
    # Labels is a list in MALT format
    labels_raw = data.get("labels", [])
    if isinstance(labels_raw, str):
        labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
    else:
        labels = [str(l).strip() for l in labels_raw if l]
    is_hack = bool(set(labels) & MALT_HACK_LABELS)
    text = extract_text_from_malt(data)
    if len(text.strip()) < 50:
        return None
    return {
        "id": f"malt_{filepath.stem}",
        "source": "malt",
        "text": text,
        "is_hack": is_hack,
        "hack_type": next((l for l in labels if l in MALT_HACK_LABELS), None),
        "task_family": data.get("task_family", "unknown"),
    }


def extract_text_from_arb(data):
    text_parts = []
    for key in ["trajectory", "actions", "steps", "content", "text"]:
        if key in data:
            if isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        text_parts.extend(str(v) for k, v in item.items() if v)
                    else:
                        text_parts.append(str(item))
            else:
                text_parts.append(str(data[key]))
    return "\n".join(text_parts)


def convert_arb(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
    except:
        return None
    side_effects = data.get("side_effects", False)
    text = extract_text_from_arb(data)
    if len(text.strip()) < 50:
        return None
    return {
        "id": f"arb_{filepath.stem}",
        "source": "arb",
        "text": text,
        "is_hack": side_effects,
        "hack_type": "side_effect" if side_effects else None,
        "task_family": data.get("benchmark", "web"),
    }


def main():
    random.seed(42)
    all_trajs = []
    malt_dir, _arb_dir = Path("data/real_trajectories/malt"), Path("data/real_trajectories/arb")
    if malt_dir.exists():
        for f in malt_dir.glob("*.json"):
            t = convert_malt(f)
            if t:
                all_trajs.append(t)
        print(f"MALT: {len([t for t in all_trajs if t['source'] == 'malt'])} converted")
    # ARB has complex nested structure - skip for now, MALT has 7k+ trajectories
    # if arb_dir.exists():
    #     for f in arb_dir.rglob("*.json"):
    #         t = convert_arb(f)
    #         if t: all_trajs.append(t)
    #     print(f"ARB: {len([t for t in all_trajs if t['source']=='arb'])} converted")
    print("ARB: skipped (complex format, MALT sufficient)")
    print(f"Total: {len(all_trajs)} | Hacks: {sum(1 for t in all_trajs if t['is_hack'])}")
    random.shuffle(all_trajs)
    n = len(all_trajs)
    train, val, test = (
        all_trajs[: int(n * 0.7)],
        all_trajs[int(n * 0.7) : int(n * 0.85)],
        all_trajs[int(n * 0.85) :],
    )
    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(data, f)
        print(f"{name}: {len(data)} ({sum(1 for t in data if t['is_hack'])} hacks)")


if __name__ == "__main__":
    main()
