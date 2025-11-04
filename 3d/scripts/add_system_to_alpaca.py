import argparse, json, os, sys

SYSTEM_TEXT = "You are an expert in causal inference and graph theory."

def load_any(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    # 尝试 JSON 数组
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data, "json"
    except Exception:
        pass
    # 尝试 JSONL
    items = []
    for i, line in enumerate(txt.splitlines()):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        try:
            obj = json.loads(s)
            items.append(obj)
        except Exception as e:
            print(f"跳过无法解析的第{i+1}行: {e}", file=sys.stderr)
    if items:
        return items, "jsonl"
    raise ValueError("无法解析为 JSON 或 JSONL")

def write_any(items, out_path, mode_hint=None):
    is_jsonl = mode_hint == "jsonl" or out_path.lower().endswith(".jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        if is_jsonl:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        else:
            json.dump(items, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=False,
                    default="/opt/data/private/FYP33_OYXX/_HypoSpace/3d/datasets/3d_finetune_alpaca_fixed.json")
    ap.add_argument("--out", dest="out", required=False,
                    default="/opt/data/private/FYP33_OYXX/_HypoSpace/3d/datasets/3d_finetune_alpaca_fixed_with_system.json")
    args = ap.parse_args()

    items, mode = load_any(args.inp)
    fixed = []
    for obj in items:
        if isinstance(obj, dict):
            obj["system"] = SYSTEM_TEXT
        fixed.append(obj)
    write_any(fixed, args.out, mode_hint=mode)
    print(f"已处理 {len(fixed)} 条，输出到: {args.out}（模式: {mode}）")

if __name__ == "__main__":
    main()