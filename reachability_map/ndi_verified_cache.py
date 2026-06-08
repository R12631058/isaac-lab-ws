import json
import math
import os
import tempfile
from datetime import datetime, timezone


DEFAULT_CACHE_PATH = os.path.join(
    "scripts", "isaaclab_ws", "reachability_map", "ndi_verified_runs.jsonl"
)


def default_entry_pos(tumor_pos, tumor_angle):
    theta_rad = math.radians(float(tumor_angle or 0.0))
    return [
        float(tumor_pos[0]),
        float(tumor_pos[1]) - 0.10 * math.tan(theta_rad),
        float(tumor_pos[2]) + 0.10,
    ]


def round_pos(pos, digits=3):
    return [round(float(v), digits) for v in pos]


def build_key(tumor_pos, entry_pos, tumor_angle, block_id, base_x, pos_digits=3):
    return {
        "tumor_pos": round_pos(tumor_pos, pos_digits),
        "entry_pos": round_pos(entry_pos, pos_digits) if entry_pos is not None else None,
        "tumor_angle": round(float(tumor_angle or 0.0), 1),
        "block_id": str(block_id),
        "base_x": round(float(base_x), 3),
    }


def key_id(key):
    return "|".join([
        str(key["block_id"]),
        f"{key['base_x']:.3f}",
        ",".join(f"{v:.3f}" for v in key["tumor_pos"]),
        ",".join(f"{v:.3f}" for v in key["entry_pos"]) if key["entry_pos"] is not None else "-",
        f"{key['tumor_angle']:.1f}",
    ])


def load_records(cache_path=DEFAULT_CACHE_PATH):
    if not os.path.exists(cache_path):
        return []
    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[NDI Cache][WARN] Skipping malformed line {line_no}: {exc}")
    return records


def find_latest(records, tumor_pos, entry_pos, tumor_angle, block_id, base_x):
    wanted = build_key(tumor_pos, entry_pos, tumor_angle, block_id, base_x)
    wanted_id = key_id(wanted)
    for record in reversed(records):
        record_key = record.get("key", {})
        try:
            record_id = key_id(record_key)
        except (KeyError, TypeError, ValueError):
            continue
        if record_id == wanted_id:
            return record
    return None


def append_record(cache_path, record):
    cache_path = os.path.abspath(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    record = dict(record)
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def compact_records(cache_path=DEFAULT_CACHE_PATH):
    records = load_records(cache_path)
    latest_by_key = {}
    for record in records:
        record_key = record.get("key")
        if record_key:
            latest_by_key[key_id(record_key)] = record

    cache_path = os.path.abspath(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="ndi_verified_", suffix=".jsonl", dir=os.path.dirname(cache_path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in latest_by_key.values():
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                f.write("\n")
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
