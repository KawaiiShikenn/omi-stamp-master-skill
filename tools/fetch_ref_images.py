"""从 Wikimedia Commons 下载公版中国邮票参考图（个人本地项目用）"""
import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
CATALOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "catalog")
IMAGES_DIR = os.path.join(CATALOG_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# id -> 搜索词（只选已进入公有领域的票，1980 之后的跳过版权风险）
QUERIES = {
    "p1": "Tiananmen gate stamp 1950 China",
    "t57": "Huangshan stamp 1963 China",
    "t38": "Goldfish stamp China 1960",
    "j94": "Mei Lanfang stamp 1962 China",
    "w1": "Mao Zedong stamp 1967 China",
}


def api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "stamp-recognizer/1.0 (open-source)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def search_images(query, limit=8):
    data = api({
        "action": "query", "list": "search", "srsearch": query,
        "srnamespace": "6", "format": "json", "srlimit": limit,
    })
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def get_image_url(title, width=600):
    data = api({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size|mime", "iiurlwidth": width, "format": "json",
    })
    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        ii = p.get("imageinfo", [])
        if ii:
            info = ii[0]
            if info.get("mime", "").startswith("image/"):
                return info.get("thumburl") or info.get("url")
    return None


def main():
    result = {}
    for sid, query in QUERIES.items():
        titles = search_images(query)
        saved = None
        for t in titles:
            url = get_image_url(t)
            if not url:
                continue
            ext = ".jpg" if "jpeg" in url or ".jpg" in url else ".png"
            dest = os.path.join(IMAGES_DIR, sid + ext)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "stamp-recognizer/1.0 (open-source)"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                    f.write(r.read())
                saved = os.path.basename(dest)
                print(f"[ok] {sid} <- {t} ({url})")
                break
            except Exception as e:
                print(f"[skip] {sid} {t}: {e}")
        result[sid] = saved
    # 更新 stamps.json 的 image 字段
    meta_path = os.path.join(CATALOG_DIR, "stamps.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    for entry in catalog:
        if entry["id"] in result and result[entry["id"]]:
            entry["image"] = result[entry["id"]]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print("done:", result)


if __name__ == "__main__":
    main()
