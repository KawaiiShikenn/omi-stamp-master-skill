"""邮票区域检测、裁剪、透视矫正"""
import cv2
import numpy as np


def detect_stamp(image, debug=False):
    """检测图像中的邮票区域，返回 (矫正后的邮票图, 角点或 None)。

    思路：Canny 边缘 -> 膨胀 -> 最大外轮廓 -> 多边形逼近 -> 透视矫正。
    找不到明显轮廓时原图直出。
    """
    img = image.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * h * w:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if 4 <= len(approx) <= 8 and area > best_area:
            best_area = area
            best = approx

    if best is None:
        return image, None

    if len(best) == 4:
        pts = best.reshape(4, 2).astype("float32")
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(best)).astype("float32")

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    rect = np.array([tl, tr, br, bl], dtype="float32")

    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, rect


def imread_unicode(path):
    """跨平台读图：优先 cv2.imread（Linux/macOS 原生支持 UTF-8 路径），
    Windows 中文路径失败时退回 np.fromfile + imdecode。"""
    img = cv2.imread(str(path))
    if img is None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def load_image(path):
    """读取图片（支持中文路径，跨平台）"""
    return imread_unicode(path)


def _iou_regions(a, b):
    """两个 bbox 的 IoU（用 min 面积做分母，对包含关系敏感）"""
    x0, y0, x1, y1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(x0, bx0), max(y0, by0)
    ix1, iy1 = min(x1, bx1), min(y1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / max(min((x1 - x0) * (y1 - y0), (bx1 - bx0) * (by1 - by0)), 1)


def detect_by_ocr(image, max_side=2000, up_pad=8, down_pad=1.5):
    """OCR 引导拆图：用票面文字框定位邮票区域（Canny 失效时的 fallback）。

    邮票上必有"中国邮政/中国人民邮政"等字样，文字行上方通常就是票面主体，
    因此把文字行向上扩展 up_pad 倍高度、向下 down_pad 倍高度作为候选区域。
    大图先缩放再 OCR，坐标映射回原图。
    """
    from rapidocr_onnxruntime import RapidOCR

    h, w = image.shape[:2]
    scale = 1.0
    ocr_img = image
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        ocr_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    engine = RapidOCR()
    result, _ = engine(ocr_img)
    if not result:
        return []

    boxes = [np.array(r[0], dtype=np.float32) / scale for r in result]
    # 按 y 中心聚类文字框为行
    rows = []
    for box in boxes:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        cy = (sum(ys) / 4)
        placed = False
        for r in rows:
            if abs(cy - r["cy"]) < 60:
                r["boxes"].append(box)
                r["cy"] = (r["cy"] * len(r["boxes"]) + cy) / (len(r["boxes"]) + 1)
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "boxes": [box]})

    regions = []
    for r in rows:
        xs = [p[0] for b in r["boxes"] for p in b]
        ys = [p[1] for b in r["boxes"] for p in b]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        row_h = y1 - y0
        x0 = max(0, int(x0 - (x1 - x0) * 0.5))
        x1 = min(w, int(x1 + (x1 - x0) * 0.5))
        y0 = max(0, int(y0 - row_h * up_pad))
        y1 = min(h, int(y1 + row_h * down_pad))
        if (x1 - x0) < 0.05 * w or (y1 - y0) < 0.03 * h:
            continue
        regions.append((x0, y0, x1, y1))

    # ---- P1：细长条按邮票先验扩展 ----
    # 竖排文字（贺龙票"贺龙同志诞生九十周年"）拆图时常切出宽度远小于高度的
    # 细长条（宽高比 < 0.4）——只框住了文字带，没框住票面主体。
    # 横排票文字带扩展后宽高比通常 ~0.7，不会误触发。
    # 邮票常见版式横版 40×30（宽:高≈1.33）、竖版 30×40（宽:高≈0.75），
    # 按中心把宽度扩展到覆盖票面，再补足高度。
    expanded = []
    for x0, y0, x1, y1 in regions:
        bw, bh = x1 - x0, y1 - y0
        # 1) 宽度不足（真正的竖排文字带/细长条）：横向扩展
        if bw < bh * 0.4:
            target_w = int(bh * 1.2)  # 覆盖横版/竖版票宽
            cx = (x0 + x1) / 2
            nx0 = max(0, int(cx - target_w / 2))
            nx1 = min(w, int(cx + target_w / 2))
            if nx1 - nx0 < bh * 0.8:  # 顶到边界仍不够 => 向有空间一侧再扩
                if nx0 == 0:
                    nx1 = min(w, nx0 + target_w)
                else:
                    nx0 = max(0, nx1 - target_w)
            x0, x1 = nx0, nx1
        # 2) 高度不足：纵向补足到宽度的 0.75（横版 40×30 的比例）
        if (y1 - y0) < (x1 - x0) * 0.5:
            need = int((x1 - x0) * 0.75)
            cy = (y0 + y1) / 2
            y0 = max(0, int(cy - need / 2))
            y1 = min(h, int(cy + need / 2))
        if (x1 - x0) < 0.05 * w or (y1 - y0) < 0.03 * h:
            continue
        expanded.append((x0, y0, x1, y1))
    regions = expanded

    # 合并垂直重叠区域
    regions.sort(key=lambda r: r[1])
    merged = []
    for rg in regions:
        if merged and abs(rg[1] - merged[-1][1]) < 0.15 * h and abs(rg[3] - merged[-1][3]) < 0.15 * h:
            m = merged[-1]
            merged[-1] = (min(m[0], rg[0]), min(m[1], rg[1]), max(m[2], rg[2]), max(m[3], rg[3]))
        else:
            merged.append(list(rg))

    # ---- IoU 去重：同一枚票的多个文字行扩展框互相重叠 ----
    # 徐霞客票案例：3 个框 IoU 0.88/1.00（同一枚票不同文字行扩出来的）→ 合并；
    # 亚运票案例：残片框 vs 完整票框 IoU 0.506 → 是不同对象，不能合。
    # 阈值 0.8：同一枚票的框通常 >0.8（互相包含），不同对象 <0.6。
    # 只保留面积最大的框，避免同一枚票被重复处理。
    dedup = []
    for rg in sorted(merged, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True):
        keep = True
        for kept in dedup:
            if _iou_regions(rg, kept) > 0.8:
                keep = False
                break
        if keep:
            dedup.append(rg)
    merged = dedup

    crops = []
    for x0, y0, x1, y1 in merged:
        crops.append((image[y0:y1, x0:x1], (x0, y0, x1, y1)))
    return crops
