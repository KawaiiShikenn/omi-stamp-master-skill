"""测试 embed_image"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import embed_image

p = r"<示例图片路径>"
print("exists:", os.path.exists(p))
try:
    v = embed_image(p)
    print("OK", v.shape, v.dtype)
except Exception:
    traceback.print_exc()
