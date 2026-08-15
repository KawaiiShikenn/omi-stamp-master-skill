"""千问 VL 视觉精读模块（DashScope OpenAI 兼容接口）

识别链路中的"精读裁判"——仅当本地 OCR/CLIP 都没把握时调用：
从邮票图上读出结构化字段（志号/票名/年份/面值/票面文字），
上层用这些字段精确检索本地目录（vision_search），千问只"读"不"编"。

- API Key 读取顺序：环境变量 DASHSCOPE_API_KEY → 项目根 .env 文件（同目录上上级）→ 未配置时返回 None（链路自动降级，不报错）
- 模型默认 qwen-vl-max（VISION_MODEL 环境变量可覆盖）
- 网络/解析失败一律返回 None，绝不阻塞识别主流程
"""
import base64
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("VISION_MODEL", "qwen-vl-max")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
TIMEOUT = 60

# 千问结果缓存：图感知 hash -> 结构化结果 JSON。同一张图/同一区域重复识别
# 不重复调 API（省云 token）。缓存目录在项目根 .vision_cache/（gitignore 排除）
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".vision_cache")
_CACHE_SIZE = 64  # 感知 hash 缩放尺寸（像素）

# 真实 API 调用计数（缓存命中不计入）。recognize() 每次识别前 reset，
# 输出 vision_calls 让用户知道本轮是否烧了云 token（成本透明）。
_api_calls = 0


def reset_api_calls():
    global _api_calls
    _api_calls = 0


def get_api_calls():
    return _api_calls


def _load_api_key():
    """环境变量优先，其次项目根 .env（vision_llm.py 位于 tools/，项目根在其上两级）"""
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        env_file = os.path.join(root, ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("DASHSCOPE_API_KEY=") and not line.startswith("#"):
                            return line.split("=", 1)[1].strip()
            except OSError:
                pass
    return ""

PROMPT = """你是邮票识别助手。仔细看这张邮票图片，提取可见信息，只输出一个 JSON 对象，不要输出任何其他文字。

JSON 字段：
{
  "catalog_no": "票面印的志号，如 J126 / T44 / 2005-12 / 个9；看不清或没有则填 null，绝不要猜测",
  "name": "票名（票面文字），如 贺龙同志诞生九十周年；看不清填 null",
  "year": "票面出现的年份数字（如 1986）；没有填 null",
  "denomination": "面值，如 0.28元 / 3.00元 / 8分；必须带单位(元/分)，不带单位的数字不是面值，填 null",
  "text_on_stamp": "票面所有能读出的文字（含艺术字/书法字/小字），逐字列出，读不出的描述字形",
  "design": "主图内容客观描述，80 字以内，尽量具体：人物（姿态/服饰/表情/是否戴眼镜）、动物（种类/颜色）、场景、色调、画面构图。这张描述会用于图案检索，越具体越好",
  "uncertain": true
}

规则：
- 只报告你实际看到的，任何字段不确定就填 null
- catalog_no 最容易看错，只有清晰可见才填
- 面值必须形如 0.08元 / 20分，数字+单位；没有单位的一律填 null
- 不要凭你的知识库补充票名，一切以票面印刷为准
- 【重点】year 字段：仔细找票面四角/边缘的所有阿拉伯数字年份（如 1986、1990），
  生肖票/纪念票通常印有发行年份小字，看到就填，这是识别关键线索；
  若票面确实没有可见年份数字才填 null"""


def _img_hash(image, size=_CACHE_SIZE):
    """图感知 hash：文件路径或 numpy 数组 -> md5 字符串。

    用于千问结果缓存。裁剪区域图是确定性变换，同图同区域 hash 稳定。
    """
    try:
        import cv2
        import numpy as np

        if hasattr(image, "ndim"):
            arr = image
        else:
            # 跨平台读图：先 cv2.imread（Linux/macOS 支持 UTF-8 路径），再退回 fromfile
            arr = cv2.imread(str(image))
            if arr is None:
                arr = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return ""
        small = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()
    except Exception:
        return ""


def _img_to_base64_dataurl(path, max_side=1280):
    """读图（文件路径 或 cv2/numpy 数组）-> 限制最长边（省视觉 token）-> JPEG base64 data URL。

    max_side 1280 + quality 85：比旧值(1600/90)省约 30-40% 视觉 token，
    对邮票票面文字识别影响可接受（测试验证）。
    """
    try:
        from PIL import Image

        if hasattr(path, "ndim"):  # numpy 数组（cv2 裁剪结果，BGR）
            import cv2

            arr = path
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(arr)
        else:
            im = Image.open(path)
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _parse_json(text):
    """从模型输出里抽出 JSON（容错 markdown code fence / 前后杂文）"""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def vision_read_stamp(image_path, model=DEFAULT_MODEL, api_key=None):
    """千问 VL 精读一枚邮票（裁剪后的单枚图）。返回 dict 或 None。

    带缓存：同图同区域重复识别直接读缓存，不重复调 API（省云 token）。
    """
    api_key = (api_key or _load_api_key()).strip()
    if not api_key:
        return None

    # ---- 缓存命中：不调 API ----
    img_hash = _img_hash(image_path)
    if img_hash:
        cache_file = os.path.join(_CACHE_DIR, img_hash + ".json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass

    data_url = _img_to_base64_dataurl(image_path)
    if not data_url:
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出 JSON，不输出其他内容。"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        global _api_calls
        _api_calls += 1  # 真实 API 请求（无论成功失败都计入，缓存命中不计）
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        obj = _parse_json(content)
        if obj:
            _sanitize_vision(obj)
            # 写缓存（仅成功结果，失败不缓存）
            if img_hash:
                try:
                    os.makedirs(_CACHE_DIR, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False)
                except OSError:
                    pass
        return obj
    except Exception:
        return None


def _sanitize_vision(obj):
    """后处理校验：拦截千问幻觉字段（面值离谱/年份非4位数字）"""
    den = str(obj.get("denomination") or "").strip()
    if den:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(元|分)", den)
        if not m:
            obj["denomination"] = None  # 不带单位的数字不是面值
        else:
            val = float(m.group(1))
            if val < 0.04 or val > 100:
                obj["denomination"] = None  # 离谱面值（如 9.85元 幻觉）
    year = str(obj.get("year") or "").strip()
    if year and not re.fullmatch(r"\d{4}", year):
        obj["year"] = None  # 年份必须是 4 位数字（"一八六八—一九八八"这类不算）


def _norm_catalog_no(s):
    """志号归一化：
    1. 删括号内容（图序，如 (2-1)）
    2. 只保留字母/数字/中文（点号/横线/空格等全去），转大写
    例：J.146.(2-1) -> J146；J.146 -> J146；2005-12 -> 200512；T.44 -> T44
    """
    s = re.sub(r"[（(][^)）]*[)）]", "", str(s or ""))
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s).upper()


def _extract_figure_no(s):
    """提取千问读出的图序，如 (6-4) -> (6, 4)。用于全套枚数校验：
    目标条目没有 6 枚票却读出 (6-4)，说明志号被误读（J83 柯棣华只有 2 枚，
    不可能有 6-4）=> 志号命中作废，防止带偏。"""
    m = re.search(r"[（(](\d+)-(\d+)[)）]", str(s or ""))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _name_consistent(vname, entry_name):
    """千问读出的票名与目录条目名是否一致（无公共中文 2-gram 即视为不一致）。

    千问可能误读志号（T.128 -> T.126）且脑补票名（"中国船舶"），
    此时志号命中的条目名与读出的票名对不上 => 志号命中不可信，降权。
    vname 为空时无从校验，视为一致。
    """
    vname = str(vname or "").strip()
    if not vname:
        return True
    ename = str(entry_name or "").strip()
    if not ename:
        return True

    def grams(s):
        segs = re.findall(r"[\u4e00-\u9fff]{2,}", s)
        return {segs[i][j:j + 2] for seg in segs for i in range(len(seg)) for j in range(len(seg[i]) - 1)}

    return bool(grams(vname) & grams(ename))


def vision_search(vision, entries, top_k=15):
    """用千问读出的字段检索目录。返回 [(命中数, entry), ...] 按命中数降序。

    证据权重：
      1. 志号（catalog_no 归一化精确/前缀匹配，+100，一条命中即一锤定音）
      2. 年份（issue_date 前缀匹配，+3）
      3. 票名/票面文字 2-gram 关键词（子串匹配，+1，过滤"中国人民邮政"等噪音）
    """
    if not vision:
        return []

    BAD_CHARS = "中国邮政人民华共"

    def norm_kw(seg):
        return not any(c in BAD_CHARS for c in seg)

    scores = {}

    def add(entry, w):
        scores[entry["id"]] = scores.get(entry["id"], 0) + w

    # 1) 志号（最强证据）：精确 +100；前缀（目录侧长度≥3） +50。
    #    票名一致性校验：千问误读志号时（T.128->T.126），命中的条目名与读出的票名
    #    不一致 => 证据降权（100->30 / 50->15），避免不相关票被一锤定音。
    #    图序校验：千问读出 (6-4) 但目标条目全套枚数 < 6
    #    => 志号必为误读（如 J93 被读成 J83，J83 柯棣华只有 2 枚），命中作废。
    cn = _norm_catalog_no(vision.get("catalog_no"))
    vname = str(vision.get("name") or "").strip()
    fig = _extract_figure_no(vision.get("catalog_no"))
    if cn and cn != "NULL":
        for e in entries:
            ec = _norm_catalog_no(e.get("catalog_no"))
            if not ec:
                continue
            stamps_n = len(e.get("stamps", []) or [])
            if fig and fig[0] > stamps_n:
                continue  # 图序分母超过全套枚数 => 志号误读，作废
            if cn == ec:
                add(e, 100 if _name_consistent(vname, e.get("name")) else 30)
            elif len(ec) >= 3 and (ec.startswith(cn) or cn.startswith(ec)):
                add(e, 50 if _name_consistent(vname, e.get("name")) else 15)
        # ---- 志号数字容错 ----
        # 千问读 J83、真实 J93：印刷体 8/9 是最典型误读。当志号匹配全部被
        # 图序校验作废时（说明读错了），对每一位数字做 0-9 替换生成变体，
        # 命中 +50 降权（志号本身读错，不能一锤定音，需年份/面值佐证）。
        if fig and all(not (fig[0] <= len(e.get("stamps", []) or []) and
                            (_norm_catalog_no(e.get("catalog_no")) == cn
                             or (len(_norm_catalog_no(e.get("catalog_no"))) >= 3
                                 and (_norm_catalog_no(e.get("catalog_no")).startswith(cn)
                                      or cn.startswith(_norm_catalog_no(e.get("catalog_no"))))))
                            ) for e in entries):
            for i, ch in enumerate(cn):
                if not ch.isdigit():
                    continue
                for d in "0123456789":
                    if d == ch:
                        continue
                    variant = cn[:i] + d + cn[i + 1:]
                    for e in entries:
                        ec = _norm_catalog_no(e.get("catalog_no"))
                        stamps_n = len(e.get("stamps", []) or [])
                        if ec == variant and (not fig or fig[0] <= stamps_n):
                            add(e, 50 if _name_consistent(vname, e.get("name")) else 15)

    # 2) 年份
    year = str(vision.get("year") or "").strip()
    if year.isdigit():
        for e in entries:
            if str(e.get("issue_date", "")).startswith(year):
                add(e, 3)

    # 2.5) 面值：千问读出 8分/0.08元，匹配 stamps 明细
    # 里的单枚面值。面值 + 年份组合能锁定套票（J93 1983年 8分 => 6-4 跳水）。
    den = str(vision.get("denomination") or "").strip()
    den_fen = None
    m_den = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(元|分)", den)
    if m_den:
        val = float(m_den.group(1))
        den_fen = val if m_den.group(2) == "分" else val * 100
    if den_fen:
        for e in entries:
            for s in e.get("stamps", []) or []:
                sd = str(s.get("denomination") or "").strip()
                try:
                    sval = float(sd)
                except ValueError:
                    continue
                if abs(sval * 100 - den_fen) < 0.5:  # 元->分 后容差 0.5 分
                    add(e, 5)
                    break

    # 2.6) 图案描述关键词：千问 design 里出现生肖动物名
    # （虎/马/牛…）+ 剪纸/年画风格词时，匹配目录 description/stamp_name。
    # 生肖票 OCR 无字 + CLIP 扎堆，这是第三证据线（虎+8分 => T107 丙寅年）。
    # 分级：目录含年号（"寅年"）是生肖票专有特征 +8；只含动物名（"虎"）容易
    # 误中（虎头帽/虎头金鱼/武术虎形），降为 +4。
    design = str(vision.get("design") or "").strip()
    if design:
        # 生肖动物词（含年号别称）
        ZODIAC = {
            "鼠": ["鼠", "子年"], "牛": ["牛", "丑年"], "虎": ["虎", "寅年"],
            "兔": ["兔", "卯年"], "龙": ["龙", "辰年"], "蛇": ["蛇", "巳年"],
            "马": ["马", "午年"], "羊": ["羊", "未年"], "猴": ["猴", "申年"],
            "鸡": ["鸡", "酉年"], "狗": ["狗", "戌年"], "猪": ["猪", "亥年"],
        }
        found = [k for k, ws in ZODIAC.items() if any(w in design for w in ws)]
        if found:
            for e in entries:
                hay = f"{e.get('name','')}|{e.get('description','')}"
                for s in e.get("stamps", []) or []:
                    hay += f"|{str(s.get('stamp_name','') or '')}"
                for z in found:
                    year_word = ZODIAC[z][1]  # 年号词如"寅年"
                    animal_word = ZODIAC[z][0]  # 动物名如"虎"
                    if year_word in hay:
                        add(e, 8)  # 年号是生肖票专有特征
                    elif animal_word in hay:
                        add(e, 4)  # 动物名易误中，降权
                    break

    # 3) 票名/票面文字 2-gram 关键词
    text = "".join([
        str(vision.get("name") or ""),
        str(vision.get("text_on_stamp") or ""),
    ])
    segs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    kws = set()
    for seg in segs:
        for i in range(len(seg) - 1):
            g = seg[i:i + 2]
            if norm_kw(g):
                kws.add(g)
    for kw in kws:
        for e in entries:
            hay = f"{e.get('name','')}|{e.get('catalog_no','')}|{e.get('description','')}"
            if kw in hay:
                add(e, 1)

    by_id = {e["id"]: e for e in entries}
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], len(by_id[kv[0]].get("name", ""))),
    )
    return [(n, by_id[eid]) for eid, n in ranked[:top_k] if n > 0]
