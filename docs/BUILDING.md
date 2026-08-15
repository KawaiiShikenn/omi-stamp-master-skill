# 目录库构建指南

识别依赖本地目录库（`catalog/catalog.json` 元数据 + `catalog/index.npy` 图片索引），
本仓库**不附带数据**，需自行获取《中国邮票电子目录》CHM 数据源后构建。
本指南给出完整、可操作的流程。

## 数据源是什么

《中国邮票电子目录》是集邮爱好者社区整理的公开电子目录（CHM 格式），
覆盖 1949 年至今的纪特票、普票、个性化票等（约 2,003 套，含元数据与扫描图）。

- **获取方式**：通过搜索引擎检索「中国邮票电子目录 CHM」即可找到公开传播版本。
  请自行确认所获取渠道的合规性。
- **版权**：目录内容版权归原目录作者所有。本仓库不附带、不托管、不提供下载；
  仅提供解析工具与识别代码。请确认你的获取与使用方式符合当地法律及授权要求。

## 一、准备数据源

### 1. 获取 CHM 文件

用搜索引擎找到《中国邮票电子目录》CHM 下载后，你会得到若干 CHM 文件
（不同版本可能按年代/专题拆分为多个 CHM，如「JT 票」「纪特票」「普票」
「个性化」等）。

### 2. 解压 CHM

CHM 是微软帮助文档格式，解压方式任选其一：

- **Windows**：7-Zip 直接解压（右键 CHM → 7-Zip → 解压到当前目录）
- **Windows**：`hh.exe -decompile <输出目录> <文件.chm>` 命令行
- **Linux/macOS**：`7z x 文件.chm` 或 `archmage 文件.chm 输出目录`

把多个 CHM 都解压到**同一个根目录**下（各 CHM 会形成独立子目录），
例如：

```
data/
  CNJT/          # 某 CHM 解压结果
  CNCSWN/
  CNZYL/
  CN1992-2003/
  ...
```

> 子目录名随意，只要不以下划线 `_` 开头即可（`_extracted` 这类会被自动跳过）。

## 二、一键构建（推荐）

```bash
# 安装依赖（首次）
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 一键构建：自动探测来源目录 + 解析 + 建索引
python tools/setup.py <CHM解压根目录>
```

`setup.py` 会依次执行：

| 阶段 | 脚本 | 产出 |
|---|---|---|
| A. 解析 | `build_catalog.py` | `catalog/catalog.json`（2,003 套票元数据） |
| B. 索引 | `build_index.py` | `catalog/index.npy`（8,946 张图 CLIP 嵌入）+ `index.json` |

B 阶段首次运行需加载 CLIP 模型并嵌入全部目录图片，**耗时较长**
（CPU 约 30 分钟 ~ 1 小时），属正常现象，只需跑一次。

## 三、分步构建（可选）

如果你希望分步执行 / 手动控制：

```bash
# A 阶段：解析 HTML -> catalog.json
#   <data_root> 是 CHM 解压根目录
#   --sources 可指定子集（逗号分隔），默认全部
python tools/build_catalog.py <data_root> catalog/catalog.json

# B 阶段：图片嵌入 -> index.npy
python tools/build_index.py catalog/catalog.json
```

`build_index.py` 支持调试参数：

```bash
python tools/build_index.py catalog/catalog.json --limit 50   # 只处理前 50 张（调试）
python tools/build_index.py catalog/catalog.json --resume     # 跳过已嵌入的图片（中断后续跑）
```

## 四、验证构建结果

```bash
# 自检：索引是否完整
python tools/self_test.py

# 试识别一张票（零云成本路径）
python tools/recognize.py <你的邮票照片> --catalog catalog
```

预期输出：检测到邮票区域 → 完整字段（志号/名称/日期/面值/齿孔/版别…）+ Top 候选。

## 五、常见问题

**Q: `build_catalog.py` 报错 `ModuleNotFoundError: bs4`**
A: 缺少依赖，执行 `pip install -r requirements.txt`。

**Q: A 阶段解析出 0 个条目 / 大量 MISSING 图片**
A: 检查解压目录结构：`data_root` 应指向包含各 CHM 子目录的**根目录**，
   而不是某个 CHM 的内部目录；确认 HTML 里的图片相对路径存在。

**Q: B 阶段很慢 / 内存不足**
A: CLIP 嵌入是 CPU 密集任务，属正常。`--limit` 可先小规模验证；
   内存不足可分批（`--resume` 支持断点续跑）。

**Q: 没有 CHM 数据源，想先体验一下**
A: 可先用任意目录测试脚本流程，或用少量公开的邮票图片自行构造
   最小目录（参考 `catalog/stamps.json` 的字段结构）。
