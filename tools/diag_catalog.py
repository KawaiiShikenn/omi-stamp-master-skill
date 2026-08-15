"""A 阶段质量诊断"""
import json
import os
from collections import Counter

# 实际非 vti 页面数
for src in ["CN1992-2003", "CN2004-2015", "CN2016-2025", "CNCSWN", "CNJT", "CNTSYZP", "CNZYL"]:
    root = rf"<数据根目录>\{src}"
    n = 0
    for dp, dn, fn in os.walk(root):
        if "_vti_cnf" in dp:
            continue
        n += sum(1 for f in fn if f.lower().endswith((".htm", ".html")))
    print(f"{src}: 非vti页面={n}")

cat = json.load(open("catalog/catalog.json", encoding="utf-8"))
ents = cat["entries"]
print()
print("总条目:", len(ents))
print("按来源:", dict(Counter(e["source"] for e in ents)))
print("空志号:", sum(1 for e in ents if not e["catalog_no"]))
print("无日期:", sum(1 for e in ents if not e["issue_date"]))
print("专题页(两岸四地):", sum(1 for e in ents if "两岸四地" in e["name"]))
print("简介为空:", sum(1 for e in ents if not e["description"]))
print("简介异常(含'相关资料'):", sum(1 for e in ents if "相关资料" in e["description"]))
print("志号含名称(超长):", sum(1 for e in ents if len(e["catalog_no"]) > 15))
