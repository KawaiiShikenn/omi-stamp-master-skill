"""目录库：读取 catalog.json、加载预构建的 CLIP 索引、相似度匹配"""
import json
import os

import numpy as np

from embedder import cosine_sim  # noqa: F401 (兼容旧接口)

# 索引缓存：同一进程内 index.npy（约 18MB）只读盘一次。
# 之前 match_image/clip_text_search 每次调用都重新 np.load，重复 IO 浪费。
_INDEX_CACHE = {}


def load_catalog(catalog_path):
    """catalog.json -> {"entries": [...], "data_root": str, ...}"""
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(base_dir):
    """加载 build_index.py 产出的索引。返回 (Nx512 矩阵, meta 列表)。

    模块级缓存：同一进程内只读盘一次，后续调用直接复用内存矩阵。
    注意：索引重建（build_index.py）后需重启进程才生效。
    """
    if base_dir not in _INDEX_CACHE:
        mat = np.load(os.path.join(base_dir, "index.npy"))
        with open(os.path.join(base_dir, "index.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        _INDEX_CACHE[base_dir] = (mat, meta)
    return _INDEX_CACHE[base_dir]


def match_image(query_vec, base_dir, top_k=10, min_score=0.15):
    """图片级匹配：返回 [(score, meta), ...]"""
    mat, meta = load_index(base_dir)
    if mat.shape[0] == 0:
        return []
    scores = mat @ query_vec
    order = np.argsort(-scores)[:top_k]
    out = []
    for i in order:
        s = float(scores[i])
        if s >= min_score:
            out.append((s, meta[i]))
    return out


def match_entry(query_vec, base_dir, top_k=3, min_score=0.15):
    """条目级匹配：图片级结果按 entry_id 聚合，取每套票最高分。

    返回 [(score, entry_meta, matched_image_meta), ...]
    """
    results = match_image(query_vec, base_dir, top_k=50, min_score=min_score)
    best = {}
    for s, m in results:
        key = m["entry_id"]
        if key not in best or s > best[key][0]:
            best[key] = (s, m)
    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return [(s, m, m) for s, m in ranked]
