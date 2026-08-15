"""CLIP 图像编码与相似度（transformers, CPU）"""
import os

# 国内网络 huggingface.co 不可达：强制离线加载本地缓存，避免启动时联网检查卡死
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 关闭进度条/日志噪音：否则 stderr 的 "Loading weights" 进度条会让 PowerShell 管道误报 Exec failed
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor


def _extract(feats):
    import torch
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    elif hasattr(feats, "image_embeds"):
        feats = feats.image_embeds
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def embed_image(image):
    """单张：image: 文件路径 / PIL.Image / BGR ndarray -> 归一化特征向量"""
    import torch
    from PIL import Image

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1]).convert("RGB")
    model, processor = get_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = _extract(model.get_image_features(**inputs))
    return feats.squeeze(0).numpy()


def embed_images(paths):
    """批量：paths -> Nx512 归一化矩阵（CPU 批量比逐张快数倍）"""
    import torch
    from PIL import Image

    model, processor = get_model()
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = _extract(model.get_image_features(**inputs))
    return feats.numpy()


def embed_text(texts):
    """文本 -> Nx512 归一化向量（CLIP 文本编码）。

    用于「无文字票」的图案描述检索：千问读出票面图案描述（如
    "戴眼镜穿西装的人物肖像"），编码成文本向量去图库找视觉相似的票。
    """
    import torch

    model, processor = get_model()
    if isinstance(texts, str):
        texts = [texts]
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        feats = _extract(model.get_text_features(**inputs))
    return feats.numpy()


def cosine_sim(a, b):
    return float(np.dot(a, b))
