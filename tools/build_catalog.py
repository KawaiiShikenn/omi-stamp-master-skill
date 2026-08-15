"""A 阶段：把 7 个 CHM 解压出的 HTML 解析成结构化 catalog.json

用法: python build_catalog.py <data_root> <out_json> [--sources a,b,c]
data_root 示例: <CHM 解压根目录>/_extracted
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

from bs4 import BeautifulSoup

# 已知字段标签 -> 统一字段名
LABEL_MAP = {
    "志号及名称": "catalog_no",
    "全套册数": "sheet_count",
    "全套枚数": "stamp_count",
    "邮票面值": "denomination",
    "面值": "denomination",
    "售价": "price",
    "发行日期": "issue_date",
    "邮票设计": "designer",
    "设计者": "designer",
    "小本设计": "booklet_design",
    "小本规格": "size",
    "邮票规格": "size",
    "票规格": "size",
    "邮票版别": "printing",
    "版别": "printing",
    "印制机构": "printer",
    "发行机构": "issuer",
    "齿孔度数": "perforation",
    "整版枚数": "sheet_layout",
    "发行量": "mintage",
    "防伪方式": "security",
    "责任编辑": "editor",
    "摄影者": "photographer",
    "资料提供": "source_provider",
    # 图序表列头（匹配前会先归一化去空白）
    "图序": "stamp_no",
    "票图名称": "stamp_name",
    "名称": "stamp_name",
    "面值(元)": "denomination",
    "票规格(mm)": "size",
    "发行量(万)": "mintage",
}


def norm_label(s):
    """标签归一化：去掉所有空白（含全角空格），如 '设 计 者' -> '设计者'"""
    return re.sub(r"[\s\u3000]+", "", s or "")


# 归一化后的标签 -> 字段名 查找表
LABEL_NORM = {norm_label(k): v for k, v in LABEL_MAP.items()}

STAMP_TABLE_HEADERS = ["图序", "票图名称", "名称", "面值", "面值(元)", "票规格", "票规格(mm)", "齿孔度数", "发行量", "发行量(万)"]


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def decode(raw: bytes) -> str:
    for enc in ("gbk", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def clean_catalog_no(value, name):
    """志号清理：去掉混入的票名，只留编号 token（如 BPC1、1992-1、纪94）"""
    v = clean(value)
    if name:
        if v.startswith(name):
            v = v[len(name):].strip()
        elif name in v:
            v = v.replace(name, "").strip()
    parts = v.split()
    return parts[0] if parts else v


def extract_desc(soup):
    """更精准地提取“相关资料”段落"""
    label_el = soup.find(string=lambda s: s and s.strip() == "相关资料")
    if label_el:
        cell = label_el.find_parent("td")
        if cell:
            text = clean(cell.get_text(" ", strip=True))
            text = text.replace("相关资料", "", 1).strip()
            # 截掉可能混入的表头文字
            for marker in ("志号及名称", "图 序", "图序", "本 册"):
                idx = text.find(marker)
                if idx >= 0:
                    text = text[:idx].strip()
            return text
    return ""


def row_pairs(tds):
    """把 td 文本列表按 标签/值 成对拆开（支持 2/4 格行）"""
    pairs = []
    i = 0
    while i + 1 < len(tds):
        label = clean(tds[i])
        value = clean(tds[i + 1])
        if label and len(label) <= 20 and value:
            pairs.append((label, value))
        i += 2
    return pairs


def parse_page(path, source):
    with open(path, "rb") as f:
        html = decode(f.read())
    soup = BeautifulSoup(html, "html.parser")

    title = clean(soup.title.get_text()) if soup.title else ""
    m = re.match(r"^(\S+)\s+(.*)$", title)
    pid, pname = (m.group(1), m.group(2)) if m else ("", title)

    # 卷名（页面头部：中华人民共和国邮票卷 ... 小本票(1980)）
    series = ""
    for td in soup.find_all("td"):
        t = clean(td.get_text(" ", strip=True))
        if "邮票卷" in t and len(t) < 60:
            series = t
            break

    # 图片
    images = []
    base = os.path.dirname(path)
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if "photo" in src.lower() or src.lower().endswith((".jpg", ".jpeg")):
            p = os.path.normpath(os.path.join(base, src))
            if os.path.exists(p):
                rel = os.path.relpath(p, os.path.dirname(os.path.dirname(path)))
                images.append(rel.replace("\\", "/"))
            else:
                images.append(("MISSING:" + src).replace("\\", "/"))

    # 信息字段 + 图序表
    fields = {}
    extra = {}
    stamps = []
    stamp_header = None
    desc = ""
    desc_done = False
    pending_labels = None  # 标签行/值行分离布局：上一行的标签，待下一行对齐取值
    in_fdc = False  # 首日封段：字段只进 extra，不覆盖主表字段

    # 相关资料段落（用精准定位，替代行扫描）
    desc = extract_desc(soup)
    desc_done = bool(desc)

    for tr in soup.find_all("tr"):
        tds = [clean(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        tds = [t for t in tds if t]
        if not tds:
            continue
        # 嵌套表格被 find_all 拍平的大行（列数异常或单格超长）：跳过，防止污染表头/字段
        if len(tds) > 12 or any(len(t) > 300 for t in tds):
            continue

        joined = " ".join(tds)
        joined_norm = norm_label(joined)
        if "首日封" in joined:
            in_fdc = True

        # 图序表：表头行（限列数防嵌套大行污染）
        if stamp_header is None and any(h in joined for h in ("图序", "票图名称")) and len(tds) <= 12:
            stamp_header = tds
            continue
        # 图序表：数据行（跟在表头之后，行内无已知字段标签，列数须与表头接近）
        if stamp_header is not None and not any(nk in joined_norm for nk in LABEL_NORM) and abs(len(tds) - len(stamp_header)) <= 1:
            if len(tds) >= 3:
                row = {}
                for idx, h in enumerate(stamp_header):
                    if idx >= len(tds):
                        break
                    key = LABEL_NORM.get(norm_label(h))
                    if key:
                        row[key] = tds[idx]
                    else:
                        row.setdefault("extra_" + str(idx), tds[idx])
                stamps.append(row)
            continue

        # 标签行/值行分离布局（如 [齿孔度数, 发行量(万)] 下一行才是值）：记下标签，等下一行对齐
        if len(tds) >= 2 and all(norm_label(t) in LABEL_NORM for t in tds):
            pending_labels = tds
            continue
        if pending_labels:
            for idx, h in enumerate(pending_labels):
                val = tds[idx] if idx < len(tds) else ""
                key = LABEL_NORM.get(norm_label(h))
                if key:
                    fields[key] = val
                elif h not in ("",):
                    extra.setdefault(norm_label(h), val)
            pending_labels = None
            continue

        # 常规 标签/值 行（首日封段只写 extra，避免覆盖主表字段）
        for label, value in row_pairs(tds):
            key = LABEL_NORM.get(norm_label(label))
            if key and not in_fdc:
                fields[key] = value
            elif label not in ("",):
                extra.setdefault(label, value)

    # 志号清理：去掉混入的名称
    fields["catalog_no"] = clean_catalog_no(fields.get("catalog_no", pid), pname)

    # 描述兜底：<title> 后的正文
    if not desc:
        body = soup.find("body")
        if body:
            texts = [clean(t) for t in body.stripped_strings if len(clean(t)) > 20]
            desc = " ".join(texts)[:800]

    return {
        "id": pid,
        "name": pname,
        "source": source,
        "page": os.path.basename(path),
        "series": series,
        "country": "中国",
        "country_code": "cn",
        "catalog_no": fields.get("catalog_no", pid),
        "issue_date": fields.get("issue_date", ""),
        "denomination": fields.get("denomination", ""),
        "price": fields.get("price", ""),
        "designer": fields.get("designer", ""),
        "size": fields.get("size", ""),
        "printing": fields.get("printing", ""),
        "perforation": fields.get("perforation", ""),
        "printer": fields.get("printer", ""),
        "issuer": fields.get("issuer", ""),
        "mintage": fields.get("mintage", ""),
        "sheet_count": fields.get("sheet_count", ""),
        "stamp_count": fields.get("stamp_count", ""),
        "description": desc,
        "stamps": stamps,
        "images": images,
        "links": {},
        "_extra": extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", help="解压目录根（含各 CHM 子目录）")
    ap.add_argument("out_json", help="输出 catalog.json 路径")
    ap.add_argument("--sources", default="", help="逗号分隔的目录名，默认全部")
    args = ap.parse_args()

    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        sources = [d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)) and not d.startswith("_")]

    entries = []
    skipped = {"nav": 0, "no_title": 0}
    for src in sources:
        root = os.path.join(args.data_root, src)
        for dirpath, dirnames, filenames in os.walk(root):
            if "_vti_cnf" in dirpath:
                continue
            for fn in sorted(filenames):
                if not fn.lower().endswith((".htm", ".html")):
                    continue
                if fn.lower() in ("index.html", "cover.html", "desktop.ini") or "easy-chm" in fn.lower():
                    skipped["nav"] += 1
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    entry = parse_page(path, src)
                    # 专题合集页过滤：无志号且无发行日期（如“两岸四地发行的XX”汇总页）
                    if not entry["catalog_no"] and not entry["issue_date"]:
                        skipped["nav"] += 1
                        continue
                except Exception as e:
                    print(f"[error] {path}: {e}", file=sys.stderr)
                    continue
                if not entry["name"]:
                    skipped["no_title"] += 1
                    continue
                # 纯导航页过滤：无图无描述
                if not entry["images"] and not entry["description"]:
                    skipped["nav"] += 1
                    continue
                entries.append(entry)

    entries.sort(key=lambda e: (e["source"], e["id"]))
    out = {
        "version": 1,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": args.data_root,
        "count": len(entries),
        "sources": sources,
        "entries": entries,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # 统计
    img_total = sum(len(e["images"]) for e in entries)
    with_photo = sum(1 for e in entries if e["images"])
    with_date = sum(1 for e in entries if e["issue_date"])
    with_desc = sum(1 for e in entries if e["description"])
    missing_img = sum(1 for e in entries for i in e["images"] if i.startswith("MISSING"))
    print(f"== 解析完成 ==")
    print(f"来源目录: {sources}")
    print(f"有效条目: {len(entries)}（跳过导航/无标题页: {sum(skipped.values())}）")
    print(f"带图片: {with_photo}（图片总数 {img_total}，缺失 {missing_img}）")
    print(f"带发行日期: {with_date} | 带简介: {with_desc}")
    print(f"输出: {args.out_json} ({os.path.getsize(args.out_json) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
