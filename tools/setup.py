"""一键构建目录库：A 解析 CHM HTML -> catalog.json，B 图片嵌入 -> index.npy

用法:
    python tools/setup.py <CHM解压根目录> [--catalog-dir catalog]

流程:
    1. 自动探测解压根目录下的各 CHM 子目录
    2. 运行 build_catalog.py 解析 HTML -> catalog/catalog.json
    3. 运行 build_index.py 建立 CLIP 图片索引 -> catalog/index.npy + index.json

前置:
    - 已解压《中国邮票电子目录》CHM（7-Zip / chmlib 均可解压）
    - 已安装依赖（pip install -r requirements.txt + torch CPU）
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_sources(data_root):
    """探测解压目录下各 CHM 来源子目录（如 CNJT、CNCSWN、CNZYL...）"""
    if not os.path.isdir(data_root):
        print(f"[错误] 目录不存在: {data_root}", file=sys.stderr)
        sys.exit(1)
    subs = [d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith("_")]
    if not subs:
        print("[错误] 解压根目录下未找到任何子目录（CHM 解压后应有 CNJT/CNCSWN 等目录）",
              file=sys.stderr)
        sys.exit(1)
    print(f"[1/3] 探测到 {len(subs)} 个来源目录: {', '.join(sorted(subs))}")
    return subs


def step_catalog(data_root, out_json, sources):
    print(f"\n[2/3] 解析 HTML -> {os.path.relpath(out_json, ROOT)}")
    script = os.path.join(HERE, "build_catalog.py")
    cmd = [sys.executable, script, data_root, out_json, "--sources", ",".join(sources)]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("[错误] A 阶段解析失败", file=sys.stderr)
        sys.exit(r.returncode)


def step_index(catalog_json):
    print(f"\n[3/3] 建立 CLIP 图片索引（首次约需数分钟~半小时，视机器而定）")
    script = os.path.join(HERE, "build_index.py")
    cmd = [sys.executable, script, catalog_json]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("[错误] B 阶段建索引失败", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="一键构建邮票识别目录库")
    ap.add_argument("data_root", help="CHM 解压后的根目录（含 CNJT/CNCSWN 等子目录）")
    ap.add_argument("--catalog-dir", default=os.path.join(ROOT, "catalog"),
                    help="catalog.json / index.npy 输出目录（默认 ./catalog）")
    args = ap.parse_args()

    print("===== 邮票识别目录库一键构建 =====")
    sources = find_sources(args.data_root)
    os.makedirs(args.catalog_dir, exist_ok=True)
    out_json = os.path.join(args.catalog_dir, "catalog.json")

    step_catalog(args.data_root, out_json, sources)
    step_index(out_json)

    print("\n===== 构建完成 =====")
    print(f"  catalog.json: {os.path.join(args.catalog_dir, 'catalog.json')}")
    print(f"  index.npy  : {os.path.join(args.catalog_dir, 'index.npy')}")
    print(f"  index.json : {os.path.join(args.catalog_dir, 'index.json')}")
    print("\n现在可以识别了：")
    print(f"  python tools/recognize.py <图片路径> --catalog {args.catalog_dir}")


if __name__ == "__main__":
    main()
