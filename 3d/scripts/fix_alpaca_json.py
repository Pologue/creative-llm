import argparse
import json
import re
import textwrap

INSTR_FIXED = (
    "You are given observations of a 3D structure made of unit blocks on a 3x3 grid.\n"
    "Each observation shows a view of the structure from a specific angle.\n"
)

def strip_comments(text: str) -> str:
    # 去掉形如 // ... 的整行注释
    return "\n".join(
        ln for ln in text.splitlines()
        if not ln.strip().startswith("//")
    )

def split_objects(text: str):
    # 依据花括号深度切分对象，支持字符串内的换行与转义
    objs = []
    in_str = False
    escaped = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(text[start:i+1])
                        start = None
    return objs

def parse_string_literal(s: str, pos: int) -> tuple[str, int]:
    # 解析从 pos 开始的 JSON 风格字符串，允许原始换行（输入文件里的违规形式）
    assert s[pos] == '"'
    pos += 1
    out = []
    escaped = False
    while pos < len(s):
        ch = s[pos]
        if escaped:
            # 处理常见转义
            if ch == "n":
                out.append("\n")
            elif ch == "t":
                out.append("\t")
            elif ch == "r":
                out.append("\r")
            elif ch == '"':
                out.append('"')
            elif ch == "\\":
                out.append("\\")
            else:
                out.append(ch)
            escaped = False
            pos += 1
            continue
        if ch == "\\":
            escaped = True
            pos += 1
            continue
        if ch == '"':
            pos += 1
            break
        # 允许原始换行字符（修正时再正规化）
        out.append(ch)
        pos += 1
    return "".join(out), pos

def extract_field(obj_text: str, key: str) -> str | None:
    # 定位 "key": "<string>"，用自定义字符串解析器取值
    pat = re.compile(rf'"{re.escape(key)}"\s*:\s*"', re.DOTALL)
    m = pat.search(obj_text)
    if not m:
        return None
    start_quote = m.end() - 1  # 定位到开引号
    val, _ = parse_string_literal(obj_text, start_quote)
    return val

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 去共同缩进与首尾空行
    s = textwrap.dedent(s).strip("\n")
    return s

def fix_records_from_text(text: str, force_two_line_instr: bool = False):
    text = strip_comments(text)
    objs = split_objects(text)
    fixed = []
    for raw in objs:
        instr = extract_field(raw, "instruction")
        inp = extract_field(raw, "input")
        out = extract_field(raw, "output")

        instr = clean_text(instr)
        inp = clean_text(inp)
        out = clean_text(out)

        if force_two_line_instr and instr:
            # 强制替换为固定两行
            instr = INSTR_FIXED.rstrip("\n")

        # 严格保留三字段
        fixed.append({
            "instruction": instr,
            "input": inp,
            "output": out,
        })
    return fixed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="源（非标准）alpaca JSON 文件路径")
    ap.add_argument("--out", dest="out", required=True, help="输出文件路径（.json 或 .jsonl）")
    ap.add_argument("--jsonl", action="store_true", help="输出为 JSONL（默认按扩展名推断）")
    ap.add_argument("--force_two_line_instruction", action="store_true",
                    help="将 instruction 强制替换为固定两行说明")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        text = f.read()

    records = fix_records_from_text(text, force_two_line_instr=args.force_two_line_instruction)

    out_is_jsonl = args.jsonl or args.out.lower().endswith(".jsonl")
    if out_is_jsonl:
        with open(args.out, "w", encoding="utf-8") as fw:
            for r in records:
                fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(args.out, "w", encoding="utf-8") as fw:
            json.dump(records, fw, ensure_ascii=False, indent=2)

    print(f"Fixed {len(records)} records -> {args.out}")

if __name__ == "__main__":
    main()