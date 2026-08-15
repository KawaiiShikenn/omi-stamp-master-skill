"""询价模块（预留占位，v2 实现）

设计：识别模块输出 entry_id -> 本模块查 StampWorld 价格。
v2 计划：
- 用 entry 的 country_code + 志号/名称 构造 StampWorld 站内搜索
  https://www.stampworld.com/en/stamps/<country>/<...>
- 解析价格字段（注意对方站点条款与限速，先人工确认可行性）
- 输出：{entry_id, price_low, price_mid, price_high, currency, source_url, fetched_at}

v1 阶段本模块只提供接口约定，不联网。
"""


def lookup_price(entry: dict) -> dict:
    """输入 catalog.json 里的一条 entry，返回价格信息（v2 实现）"""
    raise NotImplementedError("StampWorld 询价功能计划在 v2 实现，接口已预留")


def search_url(entry: dict) -> str:
    """构造 StampWorld 搜索链接（v1 可用：识别结果里带链接，人工点开确认）"""
    country = entry.get("country_code", "china")
    name = entry.get("name", "")
    # StampWorld 站内搜索 URL 模板（v2 验证后替换为正式规则）
    return f"https://www.stampworld.com/en/stamps/{country}/?query={name}"
