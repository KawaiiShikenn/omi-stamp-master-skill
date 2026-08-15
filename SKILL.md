---
name: omi-stamp-master
description: "邮票识别：识别用户上传的邮票照片（单枚/信销票/整版多枚），返回志号、名称、发行日期、面值、齿孔、版别、发行量、简介和目录参考图。数据源为本地《中国邮票电子目录》。| Recognize Chinese stamps from photos: catalog number, name, issue date, denomination, perforation, description + reference image. Local offline catalog."
version: "1.0.0"
user-invocable: true
allowed-tools: Read, Write, Edit, Exec
---

# Omi 的邮票识别大师 / Omi's Stamp Master

识别中国邮票照片（1949-2025 全覆盖，本地离线优先）。

## 功能

- **单枚/多枚识别**：单枚照片、信销票（带邮戳）、整版/方连多枚同框均可
- **完整收藏信息**：志号、名称、发行日期、面值明细、齿孔、尺寸、版别、印刷厂、发行量、简介
- **目录参考图**：识别结果附带目录原票扫描图，供人工核对
- **成本透明**：输出 `vision_calls` 字段标记本次视觉大模型调用次数（纯本地路径零云成本）
- **三级证据链**：OCR 票面文字 → CLIP 图像匹配 → 视觉大模型精读（仅前两者无把握时触发）

## 使用流程

1. 保存用户上传的图片到临时路径（如 `tmp/input_<时间戳>.jpg`）
2. 执行识别：

   ```bash
   python tools/recognize.py <图片路径> --catalog catalog --json
   ```

3. 解析 JSON 输出，向用户呈现每枚票的结果（见输出格式）

### 常用参数

- `--json`：输出结构化 JSON（含参考图路径、`vision_calls` 成本标记）
- `--no-vision`：禁用视觉大模型精读（纯本地，零云成本）
- `--top N`：返回候选数量（默认 5）
- `--catalog <目录>`：指定目录库位置（默认 `catalog/`）

### 目录库准备

识别依赖本地目录库（`catalog/catalog.json` + `catalog/index.npy`）。
数据源为《中国邮票电子目录》CHM，获取与构建方法见 `docs/BUILDING.md`：

```bash
python tools/setup.py <CHM解压根目录>
```

## 输出格式（每枚票）

- **完整票（completeness=full）**：志号、名称、相似度（置信度）+ 完整字段
  （发行日期、面值含单枚明细、设计者、齿孔、尺寸、版别、印刷厂、发行量、简介）+ Top3 候选
- **不完整区域（completeness=partial）**：标注「不完整」+ 原因（区域过小/宽高比异常/
  与已识别区域重复/含多个志号/与区域N重叠），给 3 个候选 + 方位描述（九宫格）
- **成本标记（顶层）**：`vision_calls` = 本次实际调用视觉大模型 API 的次数（缓存命中不计）
- **证据来源 source**：`vision`（视觉精读）/ `ocr`（票面文字检索）/ `clip_text`
  （图案描述检索）/ `clip`（图案匹配）

## 识别链路

```
输入图片
  └─ 拆图（Canny 轮廓 / OCR 文字框定位 / 竖排扩展 / IoU 去重）
       └─ OCR 票面文字
            ├─ 高置信 → 文字检索目录 → 直接出结果（零云成本）
            └─ 低置信 → CLIP 图像匹配
                 ├─ 高置信 → 出结果
                 └─ 低置信/扎堆 → 视觉大模型精读（志号/年份/面值/图案）
                      └─ 多证据融合：志号 > 年份/面值/生肖 > OCR > CLIP
```

## 已知边界

- **密集册页拆图**：3 行 12+ 枚的册页照片只能拆出 1-2 枚，根治需单枚拍摄
- **黑白低对比度票**：OCR 无字 + 视觉模型也难读时诚实降级为 uncertain
- **CLIP 区分度**："个"字头红底大字个性化票分数扎堆，排名不可靠，
  靠视觉精读志号 + 人工看图确认
- **OCR 读不出书法/艺术字**，但印刷体文字可靠

## 项目结构

```
omi-stamp-master/
  tools/
    setup.py            # 一键构建目录库（解析 + 建索引）
    recognize.py        # 主入口：图片 -> 识别结果
    preprocess.py       # 邮票区域检测/裁剪/矫正/多枚拆图
    ocr_engine.py       # RapidOCR 封装
    embedder.py         # CLIP 编码与比对
    catalog.py          # 目录库加载与匹配
    build_catalog.py    # CHM 解压数据 -> catalog.json 解析器
    build_index.py      # 目录图片 -> CLIP 索引
    vision_llm.py       # 视觉大模型精读（结果缓存 + 输出校验）
  docs/
    BUILDING.md         # 目录库构建完整指南
  catalog/              # 目录库（构建生成，不随仓库分发）
```

## 版权与免责

- 目录库数据版权归原目录作者所有，不随本仓库分发（获取与构建见 docs/BUILDING.md）
- 识别结果仅供参考，不构成鉴定/估价依据
- 完整免责条款见 DISCLAIMER.md
