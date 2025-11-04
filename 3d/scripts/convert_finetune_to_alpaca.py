import argparse
import json
import sys

INSTR_FIXED = (
    "You are given observations of a 3D structure made of unit blocks on a 3x3 grid.\n"
    "Each observation shows a view of the structure from a specific angle.\n"
)

def parse_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # 跳过无法解析的行
                continue

def split_instr_input(user_content: str):
    if not isinstance(user_content, str):
        return INSTR_FIXED, ""
    # 优先精确匹配固定的 instruction 前缀
    if user_content.startswith(INSTR_FIXED):
        rest = user_content[len(INSTR_FIXED):].lstrip("\n")
        return INSTR_FIXED.rstrip("\n"), rest
    # 退化：按前两行拆分
    lines = user_content.splitlines()
    if len(lines) >= 2:
        instr = "\n".join(lines[:2])
        rest = "\n".join(lines[2:]).lstrip("\n")
        return instr, rest
    # 仅一行或更少，全部放 instruction，input 为空
    return user_content, ""

def to_alpaca_record(obj):
    # 已是 alpaca 格式则透传
    if all(k in obj for k in ("instruction", "input", "output")):
        return {
            "instruction": obj["instruction"],
            "input": obj.get("input", ""),
            "output": obj.get("output", "")
        }
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        user = next((m for m in msgs if m.get("role") == "user"), None)
        assistant = next((m for m in msgs if m.get("role") == "assistant"), None)
        user_content = user.get("content") if isinstance(user, dict) else ""
        assistant_content = assistant.get("content") if isinstance(assistant, dict) else ""
        instr, inp = split_instr_input(user_content or "")
        return {
            "instruction": instr,
            "input": inp,
            "output": assistant_content or ""
        }
    # 其他非常规结构：尝试使用 text/content 字段
    user_text = obj.get("user") or obj.get("prompt") or obj.get("input") or obj.get("content") or obj.get("text") or ""
    assistant_text = obj.get("assistant") or obj.get("output") or obj.get("response") or ""
    instr, inp = split_instr_input(user_text or "")
    return {
        "instruction": instr,
        "input": inp,
        "output": assistant_text or ""
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/opt/data/private/FYP33_OYXX/_HypoSpace/3d/datasets/3d_finetune.jsonl", help="源 JSONL (messages 格式)")
    ap.add_argument("--out", dest="out", default="/opt/data/private/FYP33_OYXX/_HypoSpace/3d/datasets/3d_finetune_alpaca.jsonl", help="目标 JSONL (alpaca 格式)")
    args = ap.parse_args()

    n_in, n_out = 0, 0
    with open(args.out, "w", encoding="utf-8") as fw:
        for rec in parse_jsonl(args.inp):
            n_in += 1
            alp = to_alpaca_record(rec)
            fw.write(json.dumps(alp, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Converted {n_out}/{n_in} records -> {args.out}")

if __name__ == "__main__":
    main()