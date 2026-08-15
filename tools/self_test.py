"""B 阶段自检：用目录库自己的图片做识别，验证索引+匹配链路"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import load_index, match_entry
from embedder import embed_image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "catalog")
mat, meta = load_index(BASE)
print(f"索引: {mat.shape}, meta: {len(meta)} 条")

# 抽样 6 张不同来源的图自检
samples = []
seen = set()
for m in meta:
    src = m["source"]
    if src in seen:
        continue
    seen.add(src)
    samples.append(m)
    if len(samples) >= 6:
        break

for m in samples:
    p = os.path.join(json.load(open(os.path.join(BASE, "catalog.json"), encoding="utf-8"))["data_root"], m["image"])
    q = embed_image(p)
    hits = match_entry(q, BASE, top_k=1, min_score=0.0)
    if hits:
        s, em, im = hits[0]
        mark = "[OK]" if em["entry_id"] == m["entry_id"] else "[FAIL]"
        print(f"{mark} 原={m['catalog_no']} {m['name'][:12]} -> 命中={em['catalog_no']} {em['name'][:12]} (score={s:.3f})")
    else:
        print(f"❌ 原={m['catalog_no']} {m['name'][:12]} -> 无命中")
