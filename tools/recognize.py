"""邮票识别主入口

用法:
    python recognize.py <图片路径> [--catalog ../catalog] [--json] [--no-vision]

流程: 读图 -> 检测矫正(多枚) -> OCR -> 文字检索优先 + CLIP 兜底 + 千问精读裁判 -> 输出结果

v2 改进 (2026-08-11):
  - OCR 文字检索优先：票面印刷体文字直接查目录名称，命中即高置信（贺龙→J126）
  - 单枚面值明细：输出每枚票的面值（如 2-1 国裕家康 1.20元 / 2-2 新春送福 3.00元），不再只用全套面值
  - 低区分度警示：Top 候选分数扎堆（差 < 0.02）时标记 low_discrimination，提示人工核对

v2.1 改进 (2026-08-12):
  - 候选全部带参考图：每个 match 含 images（目录参考图绝对路径列表），
    人工确认从"读文字猜"变成"看图秒选"（输出端附参考图）
  - 千问精读裁判：OCR/CLIP 都没把握时（OCR 无字 / CLIP 扎堆），调通义千问 VL
    精读票面（志号/票名/年份/面值/票面文字），用读出的字段查本地目录，
    千问只"读"不"编"，杜绝幻觉；无 DASHSCOPE_API_KEY 时自动降级为原流程

v2.7 提速 (2026-08-15):
  - OCR 复用：拆图阶段统一 OCR 一次，主循环不再重复 extract_text（每枚省 2-6 秒）
  - OCR 铁证跳过 CLIP：_ocr_confident 成立时不再加载 CLIP 模型+编码+全库匹配
    （有文字的票从 40-60 秒降到 ~10 秒级；CLIP 仍是无字/低置信票的兜底）
  - clip_text_search 走 load_index 缓存：不再每次 np.load 18MB 索引
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import load_catalog, load_index, match_entry
from embedder import embed_image, embed_text
from ocr_engine import extract_text
from preprocess import detect_stamp, load_image, detect_by_ocr
from vision_llm import vision_read_stamp, vision_search, reset_api_calls, get_api_calls

# 低区分度：Top1 与 Top2 分数差小于此值 => CLIP 不可信
_LOW_DISC_THRESHOLD = 0.02
# CLIP 高置信：top1 分数 >= 此值 且 与 top2 差距 >= _CLIP_GAP
_HIGH_CLIP_SCORE = 0.80
_CLIP_GAP = 0.03
# OCR 高置信：top1 命中关键词数 >= 此值 且 领先第二名
_OCR_CONFIRM_HITS = 2
# 2-gram 中包含以下字符一律丢弃（邮政/国名/单位等通用词，命中无信息量）
_BAD_CHARS = "中国邮政人民华共".strip()


def ocr_keywords(ocr_lines):
    """从 OCR 结果提取 2 字中文片段作为检索关键词（容忍 OCR 错别字）。

    逐 2-gram 提取并在字符级过滤通用字（中/国/邮/政/人/民/华/共），
    避免"中国邮政 CHINA"残留出"中国"等噪音关键词。
    """
    joined = "".join(l["text"] for l in ocr_lines)
    segs = re.findall(r"[\u4e00-\u9fff]{2,}", joined)
    kws = set()
    for seg in segs:
        for i in range(len(seg) - 1):
            g = seg[i:i + 2]
            if any(c in _BAD_CHARS for c in g):
                continue
            kws.add(g)
    return kws


def _extract_no_candidates(ocr_texts):
    """从 OCR 文本提取志号候选（含 OCR 容错：T/J 常被误读成 1，如 1.116 → T116/J116）。

    返回归一化候选列表，如 ["T116", "J116"]。
    年份误判过滤：J/T 志号数字部分最多 3 位（J 系列到
    ~J185、T 到 ~T167），4 位数字（1908）或以 19/20 开头的 3 位数字（190）
    是生卒年份（如票面"1908-1909"），不是志号。否则 190 会被当成 J90（马克思
    逝世一百周年）误伤正确答案。
    """
    joined = " ".join(ocr_texts)
    cands = set()
    for m in re.finditer(r"([1JT特纪]\.?\d{2,4})", joined):
        raw = m.group(1).replace(".", "")
        head, digits = raw[0], raw[1:]
        # 年份过滤：生卒年份 1908/1909/1990… 以 19/20 开头，
        # 而 J/T 志号数字部分最多 3 位（J 系列 ≤ ~J185、T ≤ ~T167）。
        # 必须检查完整 raw（如 "190" 是年份，不能只看 digits="90" 放过）。
        if len(digits) >= 4 or raw.startswith(("19", "20")):
            continue
        if head == "1":
            cands.add("T" + digits)
            cands.add("J" + digits)
        else:
            cands.add(head + digits)
    return sorted(cands)


def text_search(ocr_lines, entries, top_k=5):
    """OCR 文字 -> 目录名称子串检索。返回 [(命中数, entry), ...] 按命中数降序。

    志号硬匹配优先：OCR 读到志号（如 1.116.(4-1) → T116/J116）时直接加分置顶。
    志号命中必须有关键词佐证才 +100（一锤定音）；
    纯志号命中（如 OCR 把 J.136.(3-1) 误读成 155 → J55，撞上真实存在的 J055
    鉴真大师像）无文字佐证，只 +10——否则误读志号会把正确答案压下去。
    """
    kws = ocr_keywords(ocr_lines)
    no_hit = set()  # 志号命中但无任何关键词佐证的条目
    hits = {}
    # 志号硬匹配（不受 kws 为空影响）
    for cand in _extract_no_candidates([l["text"] for l in ocr_lines]):
        for e in entries:
            cno = str(e.get("catalog_no", "")).replace(".", "")
            if cno == cand:
                hits.setdefault(e["id"], [0, e])
                no_hit.add(e["id"])
    for kw in kws:
        for e in entries:
            hay = e.get("name", "") + "|" + e.get("catalog_no", "")
            # 单枚明细（stamp_name）也纳入检索：如"秦皇岛煤码头"
            for s in e.get("stamps", []):
                hay += "|" + str(s.get("stamp_name", "") or "")
            if kw in hay:
                hits.setdefault(e["id"], [0, e])[0] += 1
    # 志号加成：有关键词佐证 +100（可信，一锤定音）；无佐证 +10（可能是误读志号）
    for eid in list(no_hit):
        if eid in hits:
            hits[eid][0] += 100 if hits[eid][0] > 0 else 10
    ranked = sorted(hits.values(), key=lambda x: (-x[0], len(x[1].get("name", ""))))
    return ranked[:top_k]


def clip_text_search(description, catalog_dir, top_k=8, year_filter=None):
    """CLIP 文本检索：千问的图案描述 -> 文本嵌入 -> 图库相似度检索。

    用于「无文字票」（OCR 无字 + 千问读不出志号/票名）——
    只剩图案描述时，用 CLIP 的图文对齐能力去图库找视觉相似的票。
    返回 [(score, meta), ...]（meta 来自 index.json）。
    """
    index_path = os.path.join(catalog_dir, "index.npy")
    if not os.path.exists(index_path):
        return []
    mat, meta = load_index(catalog_dir)  # 走缓存，避免重复读盘 18MB
    qv = embed_text([description])[0]
    sims = mat @ qv
    order = np.argsort(-sims)
    out = []
    for i in order:
        out.append((float(sims[i]), meta[i]))
        if len(out) >= top_k:
            break
    return out


def denomination_detail(entry):
    """从 stamps 明细提取单枚面值列表，如 ['国裕家康 1.20元', '新春送福 3.00元']。"""
    out = []
    seen = set()
    for s in entry.get("stamps", []):
        name = str(s.get("stamp_name", "") or "")
        den = str(s.get("denomination", "") or "")
        if re.fullmatch(r"\d+\.?\d*", den) and name and "枚" not in name and "全套" not in name:
            item = f"{name} {den}元"
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _stamp_detail(entry, key):
    """从 stamps 明细提取某字段（如齿孔/尺寸/发行量）去重列表"""
    vals = []
    seen = set()
    for s in entry.get("stamps", []):
        v = str(s.get(key, "") or "").strip()
        if v and v not in seen:
            seen.add(v)
            vals.append(v)
    return vals


def _detail_str(vals, suffix=""):
    return " / ".join(v + suffix for v in vals) if vals else ""


def _build_match(src, score, full, im, data_root):
    """构造一个候选 match，带全部参考图路径（供输出端附参考图）。"""
    imgs = [os.path.join(data_root, p) for p in full.get("images", [])] if full.get("images") else []
    exact = ""
    if src == "clip" and im and im.get("image"):
        exact = os.path.join(data_root, im["image"])
    matched = exact or (imgs[0] if imgs else "")
    return {
        "score": round(score, 3) if isinstance(score, float) else score,
        "source": src,  # vision / ocr / clip
        "entry_id": full.get("id", ""),
        "catalog_no": full.get("catalog_no", ""),
        "name": full.get("name", ""),
        "country": full.get("country", "中国"),
        "country_code": full.get("country_code", "cn"),
        "year": _year_from_date(full.get("issue_date", "")),
        "issue_date": full.get("issue_date", ""),
        "theme": "",
        "denomination": full.get("denomination", ""),
        "denominations_detail": denomination_detail(full),
        "designer": full.get("designer", ""),
        "perforation": full.get("perforation") or _detail_str(_stamp_detail(full, "perforation")),
        "perforations_detail": _stamp_detail(full, "perforation"),
        "size": full.get("size") or _detail_str(_stamp_detail(full, "size")),
        "printing": full.get("printing", ""),
        "printer": full.get("printer", ""),
        "sheet_layout": full.get("sheet_layout", ""),
        "mintage_detail": _stamp_detail(full, "mintage"),
        "description": full.get("description", ""),
        "matched_image": matched,          # 首选参考图（CLIP 精确命中图或整套第一张）
        "matched_image_exact": exact,      # CLIP 精确命中的那张图（仅 clip 源）
        "images": imgs[:6],                # 整套票参考图（最多 6 张，全部给用户比对）
        "links": full.get("links", {}),
    }


def _ocr_confident(ocr_matches):
    """OCR 文字检索是否高置信：top1 命中关键词数达标且领先第二名。"""
    if not ocr_matches:
        return False
    top1 = ocr_matches[0][0]
    if top1 < _OCR_CONFIRM_HITS:
        return False
    if len(ocr_matches) >= 2 and ocr_matches[0][0] <= ocr_matches[1][0]:
        return False
    return True


def _clip_confident(clip_matches):
    """CLIP 是否高置信：top1 分高且与第二名拉开差距。"""
    if not clip_matches:
        return False
    top1 = clip_matches[0][0]
    if top1 < _HIGH_CLIP_SCORE:
        return False
    if len(clip_matches) >= 2 and (top1 - clip_matches[1][0]) < _CLIP_GAP:
        return False
    return True


def _shape_issues(bbox, img_shape):
    """形状粗筛：只查几何特征，不依赖 OCR/CLIP。返回原因列表（空 = 形状正常）。

    用于 token 优化——形状明显异常的区域（细长条/巨框/碎块）没有资格走
    CLIP+千问全链路，只配 OCR 简检给候选，省下千问的云 token。
    """
    h, w = img_shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    area_ratio = (bw * bh) / max(h * w, 1)
    aspect = bw / bh
    reasons = []
    aspect_bad = aspect > 3.0 or aspect < 1 / 3.0
    if area_ratio < 0.02:
        reasons.append(f"区域过小(约{area_ratio*100:.1f}%图幅)")
    elif area_ratio > 0.45 and aspect_bad:
        # 过大 + 宽高比异常才是页面/护邮袋框；单枚票特写（如贺龙 41.9%）形状正常不放行
        reasons.append(f"区域过大(约{area_ratio*100:.1f}%图幅，可能含多枚票)")
    if aspect_bad:
        reasons.append(f"宽高比异常({aspect:.2f})")
    return reasons


def _has_full_catalog_no(ocr_texts):
    """OCR 是否读到完整志号+图序（如 J.146.(2-2)）——形状异常但读到唯一志号的特写票可放行"""
    nos = re.findall(r"[JT特纪][.\-]?[\d]+[.\-]?[\d]*[.\-]?\(?\d-\d", " ".join(ocr_texts))
    return len(set(nos)) == 1


def recognize(image_path, catalog_dir, top_k=5, use_vision=True):
    """识别一张图（可能含多枚邮票），返回结构化结果"""
    reset_api_calls()  # 本轮千问调用计数归零（成本透明：结果带 vision_calls）
    img = load_image(image_path)
    detected = detect_all_stamps(img)  # [(stamp, bbox, ocr_lines)]

    catalog_file = os.path.join(catalog_dir, "catalog.json")
    cat = load_catalog(catalog_file) if os.path.exists(catalog_file) else None
    entry_lookup = {}
    data_root = ""
    if cat:
        for e in cat["entries"]:
            entry_lookup[e["id"]] = e
        data_root = cat.get("data_root", "")

    results = []
    seen_top = set()
    seen_boxes = []
    for i, (stamp, bbox, ocr_lines) in enumerate(detected):
        r = {
            "index": i + 1,
            "position": _position(bbox, img.shape),
            "ocr": ocr_lines,
            "matches": [],
            "low_discrimination": False,
            "vision": None,
        }
        ocr_texts = [l["text"] for l in ocr_lines]
        # ---- token 优化：形状异常（细长条/巨框/碎块）且无完整志号的区域，跳过 CLIP+千问 ----
        # 注意：is_fragment（OCR 只有邮政字样）不再触发降级——形状正常的票可能是
        # 完整票但 OCR 弱（跳水票只读出"中国人民民政"），必须放行走 CLIP+千问；
        # 残片嫌疑交给 _completeness 判 partial（不给全字段），不省链路。
        shape_reasons = _shape_issues(bbox, img.shape)
        cheap_only = bool(shape_reasons) and not _has_full_catalog_no(ocr_texts)
        if cat:
            ocr_matches = text_search(ocr_lines, cat["entries"], top_k=top_k)
            if cheap_only:
                comp, reasons = "partial", shape_reasons + ["已降级简检（省云 token）"]
                r["completeness"] = comp
                r["incomplete_reasons"] = reasons
                for hits, e in ocr_matches[:3]:
                    r["matches"].append(_build_match("ocr", hits / max(len(ocr_keywords(ocr_lines)), 1), e, None, data_root))
                results.append(r)
                continue

            query_vec = None
            # ---- v2.7：OCR 铁证（≥2 关键词且领先）时跳过 CLIP ----
            # 有文字的票直接出结果，不加载 600MB CLIP 模型、不做全库匹配。
            # CLIP 仍是 OCR 无字/低置信票的兜底。
            if _ocr_confident(ocr_matches):
                clip_matches = []
            else:
                query_vec = embed_image(stamp)
                clip_matches = match_entry(query_vec, catalog_dir, top_k=top_k)

            # 低区分度警示：CLIP Top2 分数太近 => 不可信
            if len(clip_matches) >= 2 and (clip_matches[0][0] - clip_matches[1][0]) < _LOW_DISC_THRESHOLD:
                r["low_discrimination"] = True

            # ---- 精读裁判：OCR/CLIP 都高置信才跳过千问 ----
            vision_matches = []
            clip_text_matches = []
            if use_vision and not _ocr_confident(ocr_matches) and not _clip_confident(clip_matches):
                v = vision_read_stamp(stamp)
                r["vision"] = v
                if v:
                    vision_matches = vision_search(v, cat["entries"])
                    # ---- 图案描述检索（clip_text）触发条件：志号证据不可靠 ----
                    # 千问读出志号但被图序校验作废（如 J93 误读成 J83(6-4)，J83 只有
                    # 2 枚票不可能有 6-4）=> vision_search 顶分低（志号精确命中是 100）
                    # => 必须用 design 描述去图库捞（跳水票 J93 就是靠这个救回来的）。
                    # 志号可靠（顶分 ≥ 50）时不跑，省一次 CLIP 文本嵌入。
                    top_vision = vision_matches[0][0] if vision_matches else 0
                    if top_vision < 50:
                        desc = str(v.get("design") or "").strip()
                        year = str(v.get("year") or "").strip()
                        if desc and len(desc) >= 6:
                            for s, m in clip_text_search(desc, catalog_dir, top_k=10):
                                full = entry_lookup.get(m["entry_id"], {})
                                if year.isdigit():
                                    # 年份软加权：匹配 +1.0，不硬过滤（千问可能误读年份，
                                    # 硬过滤会把正确答案杀掉，如 1985 误读成 1995）
                                    ed = str(full.get("issue_date", "") or "")
                                    if ed and ed.startswith(year):
                                        s += 1.0
                                clip_text_matches.append((s, m, full))

            # ---- 合并候选：按证据强度排序，不是固定源优先级 ----
            # 排序原则：
            #   1. vision 的弱匹配（年份 +3，如 1983 年的天鹅票）不能压过 vision 的
            #      强组合（年份+面值，如 J93 = 3+5）；按分数排即可
            #   2. clip_text（中文描述检索）对 CLIP 不可靠（英文训练，中文嵌入 0.29
            #      全是垃圾），只能放最后当兜底，绝不能压过 vision/ocr/clip
            # 排序：vision(志号≥100 一锤定音) > vision弱/ocr > clip图案 > clip_text描述
            seen_ids = set()
            ordered = []

            def _push(src, score, full, im):
                if full.get("id") in seen_ids:
                    return
                seen_ids.add(full["id"])
                ordered.append((src, score, full, im))

            for hits, e in vision_matches:
                if hits >= 100:  # 志号精确/前缀命中，最高证据
                    _push("vision", hits, e, None)
            for hits, e in vision_matches:
                if hits < 100:  # 年份/面值/关键词弱匹配，按分数降序
                    _push("vision", hits, e, None)
            for hits, e in ocr_matches:
                _push("ocr", hits / max(len(ocr_keywords(ocr_lines)), 1), e, None)
            for s, em, im in clip_matches:
                _push("clip", s, entry_lookup.get(em["entry_id"], {}), im)
            for s, m, full in clip_text_matches:
                _push("clip_text", s, full, m)

            for src, score, full, im in ordered:
                r["matches"].append(_build_match(src, score, full, im, data_root))

        # ---- 完整性判定（不完整区域只给候选，不读全字段） ----
        top_entry = r["matches"][0]["entry_id"] if r["matches"] else None
        # 千问读出有效信息（志号/年份/面值/图案任一）=> 区域含完整票面，非残片
        v_evidence = bool(r.get("vision")) and any(
            r["vision"].get(k) for k in ("catalog_no", "year", "denomination", "design", "name")
        )
        # 多枚票特征：vision 描述提到"左侧/右侧/两枚/上排/下排"等 => 区域罩住多枚票
        # （亚运票护邮袋顶框案例：千问描述"左侧邮票…右侧邮票…"，实际是两枚票的上半部）
        multi_hint = False
        if r.get("vision"):
            d = str(r["vision"].get("design") or "")
            multi_hint = any(k in d for k in ("左侧", "右侧", "左面", "右面", "两枚", "两套", "上排", "下排", "两枚票", "另一枚"))
        comp, reasons = _completeness(
            bbox, img.shape, [l["text"] for l in ocr_lines], top_entry, seen_top, seen_boxes,
            has_vision_evidence=v_evidence, multi_stamp_hint=multi_hint,
        )
        r["completeness"] = comp
        r["incomplete_reasons"] = reasons
        if top_entry:
            seen_top.add(top_entry)
        # 只有完整票（full）纳入重叠判定：partial 区域（多枚框/残片/过大）
        # 不应让后续真票因重叠而降级（亚运票 J165 被顶框拖成 partial 的回归）
        if comp == "full":
            seen_boxes.append(bbox)
        results.append(r)

    # ---- P0：整图 OCR 兜底 ----
    # 拆图失败（所有区域都无匹配）时，对原图整图 OCR 一次，
    # 用读到的票面文字走文字检索（贺龙票案例：拆图只切出细长条，
    # 但整图文字"贺龙同志诞生九十周年"可读，检索直接命中 J126）。
    # 只要已有区域给出匹配（哪怕 partial），就不触发——
    # 密集册页整图 OCR 会混读整页 10 枚票的文字，检索必然串味（J049 斯大林误报），
    # 反而污染结果。仅当全部区域都无匹配时才整图兜底。
    if cat and results and not any(r.get("matches") for r in results):
        full_ocr = extract_text(img)
        full_ocr_matches = text_search(full_ocr, cat["entries"], top_k=top_k) if full_ocr else []
        if full_ocr_matches:
            kws = ocr_keywords(full_ocr)
            denom = max(len(kws), 1)
            fr = {
                "index": len(results) + 1,
                "position": "整图",
                "ocr": full_ocr,
                "matches": [],
                "low_discrimination": False,
                "vision": None,
                "completeness": "partial",
                "incomplete_reasons": ["拆图失败，整图 OCR 兜底"],
            }
            for hits, e in full_ocr_matches[:3]:
                fr["matches"].append(_build_match("ocr", hits / denom, e, None, data_root))
            results.append(fr)
    return {"image": image_path, "stamps": results, "vision_calls": get_api_calls()}


def _position(bbox, img_shape):
    """区域中心点 -> 九宫格方位描述（相对原图）"""
    h, w = img_shape[:2]
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / max(w, 1)
    cy = (y0 + y1) / 2 / max(h, 1)
    hz = "左" if cx < 0.33 else ("右" if cx > 0.67 else "中")
    vt = "上" if cy < 0.33 else ("下" if cy > 0.67 else "中")
    names = {
        ("左", "上"): "左上角", ("中", "上"): "上方居中", ("右", "上"): "右上角",
        ("左", "中"): "左侧居中", ("中", "中"): "中央", ("右", "中"): "右侧居中",
        ("左", "下"): "左下角", ("中", "下"): "下方居中", ("右", "下"): "右下角",
    }
    return names[(hz, vt)]


def _completeness(bbox, img_shape, ocr_texts, top_entry_id, seen_entry_ids, seen_boxes=None,
                  has_vision_evidence=False, multi_stamp_hint=False):
    """判断区域是否是一枚完整邮票。返回 (full/partial, [原因])。

    不完整信号：
      1. 区域过小（占图 < 2%）：多半是碎块/切半
      2. 区域过大（占图 > 25%）：Canny 把页面/护邮袋轮廓当成邮票，可能含多枚票
      3. 宽高比异常（>3 或 <1/3）：细长条，多半切到邻票/护邮袋边
      4. 与已识别区域命中同一套票：重复区域（同一枚被切成多块）
      5. 与已识别区域 bbox 大范围重叠：区域不纯（借用邻票文字，如志号）
      6. 残片嫌疑（OCR 只有邮政字样）——千问读出有效信息时不触发（跳水票案例）
      7. 多枚票特征（vision 描述含"左侧/右侧/两枚"等，如护邮袋顶框罩住两枚票）
    """
    h, w = img_shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    area_ratio = (bw * bh) / max(h * w, 1)
    aspect = bw / bh
    reasons = []
    aspect_bad = aspect > 3.0 or aspect < 1 / 3.0
    # 区域不纯：OCR 同时读到多个不同志号（如 J.153 与 J.146 混在同一区域）
    nos = re.findall(r"[JT特纪][.\-]?[\d]+[.\-]?[\d]*[.\-]?\(?\d-\d", " ".join(ocr_texts))
    uniq_nos = sorted(set(nos))
    # 区域过大本身不代表切错：单枚票特写占图 28-40% 很常见（贺龙票 41.9%、
    # 亚运票 28.7% 都正常）。只有“过大 + 宽高比异常”或“过大 + 无票面文字”
    # 才是护邮袋/页面框（Canny 把整页当成邮票）的典型特征。
    if area_ratio < 0.02:
        reasons.append(f"区域过小(约{area_ratio*100:.1f}%图幅)")
    elif area_ratio > 0.45:
        # 超 45% 图幅：单枚票特写极限约 42%（贺龙），再大必是多枚/整页（册页案例
        # bbox 45.6% 含 2+ 枚票的 OCR 文字）=> 无条件判 partial，提示单枚拍摄
        reasons.append(f"区域过大(约{area_ratio*100:.1f}%图幅，可能含多枚票)")
    elif area_ratio > 0.25 and aspect_bad:
        reasons.append(f"区域过大(约{area_ratio*100:.1f}%图幅，可能含多枚票)")
    if aspect_bad:
        reasons.append(f"宽高比异常({aspect:.2f})")
    if multi_stamp_hint:
        reasons.append("区域含多枚票（vision 描述出现多枚特征）")
    if top_entry_id and top_entry_id in seen_entry_ids:
        reasons.append("与已识别区域重复")
    # 几何重叠：与已识别区域 bbox 重叠 > 15% 自身面积 => 区域不纯（可能借用了邻票的志号文字）
    if seen_boxes:
        for n, (bx0, by0, bx1, by1) in enumerate(seen_boxes, 1):
            ix0, iy0 = max(x0, bx0), max(y0, by0)
            ix1, iy1 = min(x1, bx1), min(y1, by1)
            if ix0 < ix1 and iy0 < iy1:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / max(bw * bh, 1) > 0.15:
                    reasons.append(f"与区域{n}重叠({inter/max(bw*bh,1)*100:.0f}%)")
                    break
    if len(uniq_nos) >= 2:
        reasons.append(f"区域含多个志号({','.join(uniq_nos)[:24]})")
    # 残片嫌疑：形状正常但 OCR 只有邮政字样/面值（无票面关键词且无完整志号）
    # 如上一轮"中国人民邮政"残片被判 full 的错误输出
    # 千问读出有效信息（图案/年份/面值）说明区域含完整票面，
    # 不是残片（跳水票 OCR 只有"中国人民民政"但千问读出红色泳衣跳水运动员）
    if not reasons and not uniq_nos and not _has_meaningful_text(ocr_texts) and not has_vision_evidence:
        reasons.append("OCR 无票面信息（残片嫌疑）")
    if not reasons:
        return "full", []
    # 即使形状可疑，只要 OCR 读到唯一完整志号+图序（如 J.146.(2-2)），仍视为完整票特写。
    # 注意："重复"必须用子串检查（reasons 元素是"与已识别区域重复"，不是"重复"）
    full_no = len(uniq_nos) == 1 and not any("重复" in r or "重叠" in r for r in reasons)
    if full_no:
        return "full", []
    return "partial", reasons


def _has_meaningful_text(ocr_texts):
    """OCR 文本里是否有票面信息关键词（排除"中国人民邮政"类噪音）。

    残片嫌疑判定用：只有邮政字样/面值数字的 OCR（如"中国人民邮政 CHINA 120"），
    说明区域可能只是邻票残片，不是完整票面。
    """
    noise = "中国邮政人民华共"
    joined = "".join(ocr_texts)
    segs = re.findall(r"[\u4e00-\u9fff]{2,}", joined)
    for seg in segs:
        for i in range(len(seg) - 1):
            g = seg[i:i + 2]
            if not any(c in noise for c in g):
                return True
    return False


def _iou(a, b):
    """两个 bbox 的 IoU（用 min 面积做分母，对包含关系敏感）"""
    x0, y0, x1, y1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(x0, bx0), max(y0, by0)
    ix1, iy1 = min(x1, bx1), min(y1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / max(min((x1 - x0) * (y1 - y0), (bx1 - bx0) * (by1 - by0)), 1)


def detect_all_stamps(image):
    """把单枚检测推广为多枚：迭代抠出邮票区域，直至无新区域（上限 12 枚）。

    返回 [(矫正后的邮票图, 原图 bbox (x0,y0,x1,y1), OCR 文字列表), ...]
    OCR 在拆图阶段统一算一次并随结果返回，主循环直接复用，避免同一区域 OCR 两遍。

    Canny 检测失败（无区域或区域 OCR 全空）时，回退到 OCR 文字框定位。
    改进（2026-08-13）：
      - 重复区域合并：掩膜残留导致同一区域反复检出，IoU>0.7 直接跳过
      - 密集册页切换：Canny 只拆出 1 枚但 OCR 定位能拆出 >=2 枚时，用 OCR 定位
    """
    import cv2

    h, w = image.shape[:2]
    stamps = []
    work = image.copy()
    for _ in range(12):
        stamp, rect = detect_stamp(work)
        if rect is None:
            break
        x0, y0 = int(rect[:, 0].min()), int(rect[:, 1].min())
        x1, y1 = int(rect[:, 0].max()), int(rect[:, 1].max())
        bbox = (x0, y0, x1, y1)
        pts = rect.astype(np.int32)
        cv2.fillPoly(work, [pts], (255, 255, 255))
        # 与已有区域高度重叠 => 掩膜残留的重复检出，丢弃
        if any(_iou(bbox, b) > 0.7 for _, b, _ in stamps):
            continue
        ocr_lines = extract_text(stamp)
        stamps.append((stamp, bbox, ocr_lines))

    if stamps:
        # 区域能读出文字 => 拆图可信；否则大概率切错，改用 OCR 定位
        has_text = any(ocr for _, _, ocr in stamps)
        if has_text:
            ocr_stamps = detect_by_ocr(image)
            # 密集册页：Canny 只找到 1 枚（重复检出合并后），但 OCR 定位能拆出多枚
            # => 说明 Canny 漏拆，用 OCR 定位结果（单枚拍摄不受影响：OCR 也是 1 个区域）
            if len(stamps) == 1 and len(ocr_stamps) >= 2:
                return [(s, b, extract_text(s)) for s, b in ocr_stamps]
            return stamps

    ocr_stamps = detect_by_ocr(image)
    if ocr_stamps:
        return [(s, b, extract_text(s)) for s, b in ocr_stamps]
    return stamps or [(image, (0, 0, w, h), extract_text(image))]


def _year_from_date(d):
    m = __import__("re").match(r"(\d{4})", d or "")
    return m.group(1) if m else ""


def render_markdown(result):
    lines = [f"## 邮票识别结果", f"- 图片: `{result['image']}`", f"- 检测到邮票区域: {len(result['stamps'])} 个"]
    vc = result.get("vision_calls", 0)
    lines.append(f"- 视觉模型(千问): {'调用 ' + str(vc) + ' 次' if vc else '未调用（纯本地，零云成本）'}")
    full_n = sum(1 for r in result["stamps"] if r["completeness"] == "full")
    lines.append(f"- 完整票: {full_n} 枚，不完整区域: {len(result['stamps']) - full_n} 个")
    for r in result["stamps"]:
        lines.append(f"\n### 区域 {r['index']}（图片{r['position']}）")
        if r["completeness"] == "partial":
            lines.append(f"- ⚠️ **不完整区域**：{'；'.join(r['incomplete_reasons'])}，不读全字段，仅给候选供参考")
            if r["matches"]:
                for m in r["matches"][:3]:
                    src_tag = {"vision": "👁️千问", "ocr": "🔤文字", "clip": "🎨图案"}.get(m["source"], m["source"])
                    lines.append(f"  - {src_tag} 候选: **{m['catalog_no']} {m['name']}**（{m['source']} {m['score']}）")
            else:
                lines.append("- 候选: 无")
            continue
        if r["ocr"]:
            lines.append("- **OCR 文字**: " + " | ".join(x["text"] for x in r["ocr"]))
        else:
            lines.append("- OCR 文字: 未识别到")
        if r.get("vision"):
            v = r["vision"]
            lines.append(
                "- **千问精读**: 志号={} 票名={} 年份={} 面值={} 票面文字={} 图案={}".format(
                    v.get("catalog_no") or "?",
                    v.get("name") or "?",
                    v.get("year") or "?",
                    v.get("denomination") or "?",
                    (v.get("text_on_stamp") or "?")[:40],
                    (v.get("design") or "?")[:40],
                )
            )
        if r["low_discrimination"]:
            lines.append("- ⚠️ 候选分数扎堆，CLIP 区分度低，请对照参考图人工确认")
        if r["matches"]:
            for idx, m in enumerate(r["matches"][:3]):
                src_tag = {"vision": "👁️千问", "ocr": "🔤文字", "clip_text": "🖼️描述检索", "clip": "🎨图案"}.get(m["source"], m["source"])
                if idx == 0:
                    denom = m['denomination'] or ' / '.join(m['denominations_detail']) or '?'
                    lines.append(
                        f"- {src_tag} **{m['catalog_no']} {m['name']}**（{m['source']} 证据分 {m['score']}）\n"
                        f"  - 发行日期: {m['issue_date']} | 面值: {denom} | 设计: {m['designer'] or '?'}\n"
                        f"  - 齿孔: {m['perforation'] or '?'} | 尺寸: {m['size'] or '?'} | 版别: {m['printing'] or '?'} | 印刷: {m['printer'] or '?'}\n"
                        f"  - 发行量: {_detail_str(m['mintage_detail'], '万枚') or '?'}\n"
                        f"  - 简介: {m['description'][:80]}\n"
                        f"  - 参考图: {m['matched_image']}"
                    )
                else:
                    lines.append(f"  - {src_tag} 候选{idx+1}: **{m['catalog_no']} {m['name']}**（{m['source']} {m['score']}）")
        else:
            lines.append("- ❌ 无匹配（可能是目录外品种，OCR 文字可作线索）")
    return "\n".join(lines)


def main():
    # Windows 控制台默认 GBK，emoji 会 UnicodeEncodeError；统一 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="邮票识别")
    ap.add_argument("image", help="邮票图片路径")
    ap.add_argument("--catalog", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "catalog"))
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--top", type=int, default=5, help="返回候选数量（默认 5）")
    ap.add_argument("--no-vision", action="store_true", help="禁用千问精读（只走本地 OCR+CLIP）")
    args = ap.parse_args()

    result = recognize(args.image, args.catalog, top_k=args.top, use_vision=not args.no_vision)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(result))


if __name__ == "__main__":
    main()
