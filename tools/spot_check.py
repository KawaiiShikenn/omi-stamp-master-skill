"""A 阶段最终抽查"""
import json

cat = json.load(open("catalog/catalog.json", encoding="utf-8"))
ents = cat["entries"]

targets = [
    ("CN1992-2003", "1992-1"),
    ("CNJT", "J001"),
    ("CNCSWN", "编001"),
    ("CNTSYZP", "T46"),
    ("CNZYL", None),
]
for src, kw in targets:
    for e in ents:
        ok = e["source"] == src and (kw is None or kw.lower() in e["catalog_no"].lower() or kw in e["id"])
        if ok:
            print(f"[{e['source']}] 志号={e['catalog_no']} | 名称={e['name']}")
            print(f"  日期={e['issue_date']} | 面值={e['denomination']} | 设计={e['designer']} | 枚数表={len(e['stamps'])} | 图={len(e['images'])}")
            print(f"  简介前60字: {e['description'][:60]}")
            break

# 图片质量抽查：尺寸
print()
print("== 图片质量抽查 ==")
from PIL import Image
import os
samples = []
for e in ents:
    for img in e["images"][:1]:
        if img.startswith("MISSING"):
            continue
        p = os.path.join(cat["data_root"], img)
        if os.path.exists(p):
            samples.append((e["catalog_no"], img, p))
            break
    if len(samples) >= 5:
        break
for no, img, p in samples:
    try:
        im = Image.open(p)
        print(f"{no} | {img} | {im.size} | {os.path.getsize(p)//1024}KB")
    except Exception as ex:
        print(f"{no} | {img} | 打开失败: {ex}")
