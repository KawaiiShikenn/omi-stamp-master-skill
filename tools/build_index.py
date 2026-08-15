"""B 阶段：为目录库图片建立 CLIP 嵌入索引（离线一次性）

用法: python build_index.py <catalog.json路径> [--limit N] [--resume]
产出:
  catalog/index.npy   Nx512 归一化嵌入矩阵
  catalog/index.json  与矩阵行对应的元数据 [{entry_id, image, page, catalog_no, name, source}]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import embed_images


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog_json", help="catalog.json 路径")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个图片（调试用）")
    ap.add_argument("--resume", action="store_true", help="跳过缓存里已有的图片")
    args = ap.parse_args()

    cat = json.load(open(args.catalog_json, encoding="utf-8"))
    entries = cat["entries"]
    base = os.path.dirname(os.path.abspath(args.catalog_json))
    cache_dir = os.path.join(base, ".emb_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 收集去重后的图片（按文件哈希去重，同一文件只嵌一次）
    seen_files, jobs = set(), []
    for e in entries:
        for img in e["images"]:
            if img.startswith("MISSING"):
                continue
            p = os.path.join(cat["data_root"], img)
            if not os.path.exists(p):
                continue
            digest = file_hash(p)
            if digest in seen_files:
                continue
            seen_files.add(digest)
            jobs.append((e, img, p, digest))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"唯一图片数: {len(jobs)}")

    vecs, metas = [], []
    t0 = time.time()
    done = 0
    batch_size = 16
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        # 过滤已缓存（resume 模式）
        todo = []
        for e, img, p, digest in batch:
            cache_file = os.path.join(cache_dir, digest + ".npy")
            if args.resume and os.path.exists(cache_file):
                vecs.append(np.load(cache_file))
                metas.append({
                    "entry_id": e["id"], "catalog_no": e["catalog_no"],
                    "name": e["name"], "source": e["source"],
                    "page": e["page"], "image": img,
                })
            else:
                todo.append((e, img, p, digest, cache_file))
        if not todo:
            continue
        try:
            batch_vecs = embed_images([t[2] for t in todo])
        except Exception as ex:
            print(f"[error] 批量失败: {ex}", file=sys.stderr)
            continue
        for (e, img, p, digest, cache_file), vec in zip(todo, batch_vecs):
            np.save(cache_file, vec)
            vecs.append(vec)
            metas.append({
                "entry_id": e["id"], "catalog_no": e["catalog_no"],
                "name": e["name"], "source": e["source"],
                "page": e["page"], "image": img,
            })
        done += len(todo)
        if done % 200 < batch_size:
            el = time.time() - t0
            print(f"  {done}/{len(jobs)}  ({el:.0f}s, {el/max(done,1)*1000:.0f} ms/张)")

    mat = np.stack(vecs)
    np.save(os.path.join(base, "index.npy"), mat)
    with open(os.path.join(base, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(metas), "meta": metas}, f, ensure_ascii=False)
    print(f"完成: {len(metas)} 张图 -> index.npy {mat.shape}，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
