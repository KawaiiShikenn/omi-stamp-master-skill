"""OCR 封装：优先 RapidOCR（onnxruntime，模型随包自带无需下载，CPU 快），easyocr 作为后备"""
import cv2
import numpy as np

_engine = None
_OCR_MAX_SIDE = 3200  # 大图先缩放再 OCR，防内存溢出；上限不宜太低，
# 否则志号/小字被缩糊（亚运票 3840px 原图能读 J.165，缩到 2400 就读丢了）


def get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def _resize_for_ocr(image):
    """超过 _OCR_MAX_SIDE 的图先等比缩放，坐标关系不影响文字识别"""
    h, w = image.shape[:2]
    if max(h, w) <= _OCR_MAX_SIDE:
        return image
    scale = _OCR_MAX_SIDE / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_text(image, min_conf=0.4):
    """image: ndarray(BGR) 或文件路径。返回 [{text, confidence}]"""
    engine = get_engine()
    if isinstance(image, np.ndarray):
        result, _ = engine(_resize_for_ocr(image))
    else:
        result, _ = engine(str(image))
    lines = []
    if result:
        for box, text, score in result:
            text = str(text).strip()
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            if text and score >= min_conf:
                lines.append({"text": text, "confidence": round(score, 3)})
    return lines


def extract_text_joined(image, min_conf=0.4):
    return " | ".join(x["text"] for x in extract_text(image, min_conf))
