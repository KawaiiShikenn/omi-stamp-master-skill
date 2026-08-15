# Contributing / 贡献指南

欢迎贡献！无论是修 bug、加功能、改进文档还是提建议，都感谢你的参与。

## 项目状态

- 本项目是一个 **OpenClaw Skill + 独立 Python 工具**，识别中国邮票（1949-2025）
- 核心链路：OCR 文字检索 → CLIP 图像匹配 → 视觉大模型精读（可选）
- 技术栈：Python 3.9+ / OpenCV / RapidOCR / CLIP (transformers) / torch

## 快速上手

```bash
# 1. 克隆并安装依赖
git clone git@github.com:kawaiishikenn/omi-stamp-master-skill.git
cd omi-stamp-master-skill
pip install -r requirements.txt

# 2. 构建目录库（识别必需）
# 数据源《中国邮票电子目录》CHM 需自行获取（版权原因不随仓库分发）
# 完整流程见 docs/BUILDING.md
python tools/setup.py <CHM解压根目录>
```

## 贡献方式

### 报告 Bug

请先搜索 [Issues](https://github.com/kawaiishikenn/omi-stamp-master-skill/issues)
确认是否已有人报告，然后用 Bug 模板提交，尽量包含：

- 运行环境（OS / Python 版本）
- 复现步骤（图片样本、命令）
- 实际输出 vs 期望输出
- 相关日志/报错信息

### 提交代码

1. Fork 本仓库，基于 `master` 新建分支（如 `fix/xxx`、`feat/xxx`）
2. 遵循现有代码风格（PEP 8，函数有 docstring，中文注释）
3. 改动尽量小且聚焦（一个 PR 解决一个问题）
4. 运行自检确认不破坏现有功能：

   ```bash
   python tools/self_test.py
   ```

5. 提交并推送到你的 Fork，然后发起 Pull Request

### PR 合并要求

- 描述清楚：改了什么、为什么改、怎么验证
- CI/自检通过（如配置了）
- 不引入敏感信息（密钥、个人路径等）

## 代码结构速览

```
tools/
  recognize.py        # 主入口（识别链路编排）
  preprocess.py       # 拆图/矫正/多枚处理
  ocr_engine.py       # RapidOCR 封装
  embedder.py         # CLIP 编码
  catalog.py          # 目录匹配（索引缓存）
  vision_llm.py       # 视觉大模型精读（可选）
  build_catalog.py    # CHM -> catalog.json
  build_index.py      # 图片 -> index.npy
  setup.py            # 一键构建
```

## 行为准则

- 友善、尊重地沟通
- 讨论技术问题就事论事，不人身攻击
- 接受不同意见，以项目最佳利益为准

## 许可

贡献即视为同意你的代码以项目 [MIT License](LICENSE) 发布。
