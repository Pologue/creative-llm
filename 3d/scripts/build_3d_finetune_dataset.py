import argparse
import json
import os
import random
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

STRICT_HEADER = "Structure:"
LAYER1_HEADER = "Layer 1:"

def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # Try JSONL
            items = []
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            return items

def normalize_whitespace_lines(s: str) -> str:
    lines = [ln.rstrip() for ln in s.strip().splitlines()]
    return "\n".join(lines).strip()

def extract_strict_structure_text(s: str) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    # Keep only from "Structure:" to end
    m = re.search(r"(?m)^Structure:\s*$", s)
    if not m:
        # Some models put 'Structure:' followed immediately by content on same line
        m = re.search(r"Structure:\s*", s)
        if not m:
            return None
    start = m.start()
    tail = s[start:].strip()
    # Ensure starts exactly with 'Structure:'
    if not tail.startswith("Structure:"):
        tail = "Structure:\n" + re.sub(r"^\s*Structure:\s*", "", tail)
    # Normalize spaces between digits to single spaces, remove commas
    fixed_lines = []
    for ln in tail.splitlines():
        if re.search(r"^\s*[\d,\s]+\s*$", ln):
            ln = ln.replace(",", " ")
            ln = " ".join(ln.strip().split())
        fixed_lines.append(ln.rstrip())
    tail = "\n".join(fixed_lines).strip()
    return tail

def parse_top_view_from_prompt(prompt: str) -> Optional[List[List[int]]]:
    # Robustly extract a 3x3 of 0/1 from the prompt text
    lines = prompt.splitlines()
    idx = None
    for i, ln in enumerate(lines):
        if "Top view" in ln:
            idx = i
    if idx is None:
        return None
    digits = []
    for ln in lines[idx+1: idx+10]:
        toks = re.findall(r"[01]", ln)
        if len(toks) == 0:
            continue
        if len(toks) >= 3:
            digits.append(list(map(int, toks[:3])))
        if len(digits) == 3:
            break
    if len(digits) != 3:
        return None
    return digits

def canonical_structure_from_top_view(top: List[List[int]]) -> Optional[str]:
    if not top or len(top) != 3 or any(len(r) != 3 for r in top):
        return None
    # Must not be all zeros for Layer 1
    if sum(sum(r) for r in top) == 0:
        return None
    out_lines = ["Structure:", "Layer 1:"]
    for r in top:
        out_lines.append(" ".join(str(x) for x in r))
    return "\n".join(out_lines)

def is_strict_format(s: str) -> bool:
    if s is None:
        return False
    s = normalize_whitespace_lines(s)
    if not s.startswith("Structure:"):
        return False
    # Minimal check: must contain Layer 1 with three rows of 3 digits
    parts = s.splitlines()
    if len(parts) < 5:
        return False
    if "Layer 1:" not in parts[1]:
        return False
    rows = parts[2:5]
    for row in rows:
        toks = row.strip().split()
        if len(toks) != 3:
            return False
        if any(t not in ("0", "1") for t in toks):
            return False
    # No trailing extra commas
    if "," in s:
        return False
    return True

def all_metrics_one(rec: Dict[str, Any]) -> bool:
    # Try common places to find metrics
    cand = []
    if isinstance(rec.get("metrics"), dict):
        cand.append(rec["metrics"])
    # Some logs may flatten metrics at top-level; filter numeric 0/1
    flat = {k: v for k, v in rec.items() if isinstance(v, (int, float)) and v in (0, 1)}
    if flat:
        cand.append(flat)
    for d in cand:
        vals = list(d.values())
        if vals and all(v == 1 for v in vals):
            return True
    return False

def get_prompt_from_record(rec: Dict[str, Any]) -> Optional[str]:
    # Common top-level text fields
    for k in ["prompt", "input", "instruction", "query", "user", "text", "content"]:
        v = rec.get(k)
        if isinstance(v, str) and "Top view" in v:
            return v
    # Sometimes under messages
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") in ("user", "system"):
                c = m.get("content")
                if isinstance(c, str) and "Top view" in c:
                    return c
    return None

def get_output_from_record(rec: Dict[str, Any]) -> Optional[str]:
    for k in ["output", "response", "completion", "assistant", "model_output", "answer", "pred", "text", "content"]:
        v = rec.get(k)
        if isinstance(v, str) and STRICT_HEADER in v:
            return v
    # Sometimes under messages
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, str) and STRICT_HEADER in c:
                    return c
    return None

def iter_results_records(results_dir: str):
    paths = sorted(glob(os.path.join(results_dir, "*.json")))
    for p in paths:
        data = load_json_any(p)
        if data is None:
            continue
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
        elif isinstance(data, dict):
            # maybe under "records" or similar
            rows = None
            for k in ["records", "items", "data", "samples", "results", "rows"]:
                v = data.get(k)
                if isinstance(v, list):
                    rows = v
                    break
            if rows:
                for rec in rows:
                    if isinstance(rec, dict):
                        yield rec

def extract_items_from_dataset_container(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ["observation_sets", "records", "items", "data", "samples", "rows", "dataset", "examples"]:
        v = data.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    # Dict-of-dicts fallback
    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        return list(data.values())
    return []

def _obs_str_to_grid9(s: str) -> Optional[List[List[int]]]:
    if not isinstance(s, str):
        return None
    s = re.sub(r"\D", "", s)  # 保底清理
    if len(s) != 9 or any(ch not in "01" for ch in s):
        return None
    nums = list(map(int, s))
    return [nums[0:3], nums[3:6], nums[6:9]]

def synth_prompt_from_item(it: Dict[str, Any]) -> Optional[str]:
    # 优先直接用已有文本
    for k in ["prompt", "input", "instruction", "query", "user", "text", "content"]:
        v = it.get(k)
        if isinstance(v, str) and "Top view" in v:
            return v

    # 支持显式 top_view
    top = it.get("top_view") or it.get("top") or it.get("grid") or it.get("matrix")
    if isinstance(top, list) and len(top) == 3 and all(isinstance(r, list) and len(r) == 3 for r in top):
        lines = [
            "You are given observations of a 3D structure made of unit blocks on a 3x3 grid.",
            "Observations (Top View - shows 1 if ANY layer has a block at that position):",
            "",
            "Top view:",
        ]
        for r in top:
            lines.append(" ".join(str(int(x)) for x in r))
        return "\n".join(lines)

    # 从 observation 的9位字符串合成 3x3 Top view
    obs = it.get("observation")
    grid = _obs_str_to_grid9(obs) if obs is not None else None
    if grid:
        lines = [
            "You are given observations of a 3D structure made of unit blocks on a 3x3 grid.",
            "Each observation shows a view of the structure from a specific angle.",
            "Observations (Top View - shows 1 if ANY layer has a block at that position):",
            "",
            "Top view:",
        ]
        for r in grid:
            lines.append(" ".join(str(x) for x in r))
        return "\n".join(lines)

    return None

def build_examples(dset_paths: List[str],
                   results_dir: Optional[str],
                   prefer_results: bool = True,
                   debug: bool = False) -> List[Dict[str, Any]]:
    # Collect perfect result outputs indexed by normalized prompt
    result_map: Dict[str, str] = {}
    if results_dir and os.path.isdir(results_dir):
        n_total, n_kept = 0, 0
        for rec in iter_results_records(results_dir):
            n_total += 1
            prompt = get_prompt_from_record(rec)
            if not prompt:
                continue
            if prefer_results and not all_metrics_one(rec):
                continue
            out = get_output_from_record(rec)
            out = extract_strict_structure_text(out) if out else None
            if out and is_strict_format(out):
                key = normalize_whitespace_lines(prompt)
                if key not in result_map:
                    result_map[key] = normalize_whitespace_lines(out)
                    n_kept += 1
        if debug:
            print(f"[DEBUG] results records scanned={n_total}, perfect_kept={n_kept}")

    examples: List[Dict[str, Any]] = []
    seen_pairs = set()

    total_items = 0
    for dp in dset_paths:
        if not os.path.isfile(dp):
            if debug:
                print(f"[DEBUG] dataset not found: {dp}")
            continue
        data = load_json_any(dp)
        if not data:
            if debug:
                print(f"[DEBUG] empty dataset: {dp}")
            continue

        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            items = extract_items_from_dataset_container(data)
        else:
            items = []

        if debug:
            print(f"[DEBUG] dataset {dp} items={len(items)} (type={type(data).__name__})")

        for it in items:
            total_items += 1
            prompt = synth_prompt_from_item(it)
            if not prompt or "Top view" not in prompt:
                if debug:
                    print(f"[DEBUG] skip item(no prompt/top view)")
                continue

            key = normalize_whitespace_lines(prompt)
            target = result_map.get(key)

            if not target:
                # Fallback to canonical from prompt
                tv = parse_top_view_from_prompt(prompt)
                if tv:
                    target = canonical_structure_from_top_view(tv)

            if not target or not is_strict_format(target):
                if debug:
                    print(f"[DEBUG] skip item(no valid target) key_hash={hash(key)%10000}")
                continue

            pair_key = (key, target)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            examples.append({
                "messages": [
                    {"role": "user", "content": key},
                    {"role": "assistant", "content": target}
                ]
            })

    # If still empty, try building purely from results (prompt+output) as a fallback
    if not examples and result_map:
        if debug:
            print("[DEBUG] fallback: building examples from results only")
        for key, target in result_map.items():
            if not is_strict_format(target):
                continue
            examples.append({
                "messages": [
                    {"role": "user", "content": key},
                    {"role": "assistant", "content": target}
                ]
            })

    if debug:
        print(f"[DEBUG] total_dataset_items={total_items}, built_examples={len(examples)}")

    return examples

def split_and_write_jsonl(examples: List[Dict[str, Any]], out_path: str, val_ratio: float = 0.05):
    random.shuffle(examples)
    n = len(examples)
    n_val = max(1, int(n * val_ratio)) if n > 10 else min(1, n)
    val = examples[:n_val]
    train = examples[n_val:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    train_path = out_path
    val_path = os.path.splitext(out_path)[0] + "_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train and {len(val)} val to:\n  {train_path}\n  {val_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "_HypoSpace/3d/datasets/3d_grid3_h3.json",
        "_HypoSpace/3d/datasets/3d_complete.json"
    ], help="Dataset JSON files to read")
    ap.add_argument("--results_dir", default="_HypoSpace/3d/results", help="Directory of benchmark result JSONs")
    ap.add_argument("--out", default="_HypoSpace/3d/datasets/3d_finetune.jsonl", help="Output JSONL path")
    ap.add_argument("--no_results", action="store_true", help="Ignore results and use canonical outputs only")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    results_dir = None if args.no_results else args.results_dir
    examples = build_examples(args.datasets, results_dir, prefer_results=True, debug=args.debug)
    examples = [e for e in examples if is_strict_format(e["messages"][1]["content"])]

    if not examples:
        print("No examples built. Check dataset/results paths and formats.")
        return

    split_and_write_jsonl(examples, args.out, val_ratio=0.05)

if __name__ == "__main__":
    main()