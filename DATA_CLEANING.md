# 数据清理方案（面向判别/评测）

本文件说明如何对 `industrial_and_scientific_items.csv` 进行清理，以便后续 judge（评测/判别）更稳定、信息更聚焦。

**数据列：**
`main_category`, `title`, `average_rating`, `rating_number`, `features`, `description`, `price`, `images`, `videos`, `store`, `categories`, `details`, `parent_asin`, `bought_together`, `subtitle`, `author`

## 1. 建议保留的必要特征
这些字段可支持“判断/对比/检索”类任务，且文本可被模型直接使用。

- `parent_asin`：唯一标识，去重与溯源必需。
- `title`：核心商品名，最重要的语义字段。
- `main_category`：大类信息，常用于分组评测。
- `categories`：细分分类，可增强语义。
- `price`：价格，用于性价比、分层等评测维度。
- `average_rating`：口碑指标。
- `rating_number`：口碑可信度（样本量）。
- `store`：品牌/店铺层面的归因分析。
- `features`：要点特征，常用于判别“是否符合需求”。
- `description`：更完整的文本描述。

## 2. 建议删除的字段（对 judge 价值低）
- `images`：通常是 URL/列表，不利于文本判别，且噪声高。
- `videos`：同上。
- `bought_together`：强依赖推荐体系，与 judge 目标弱相关。
- `subtitle`：信息增量不稳定，常为空。
- `author`：工业品类常为空或无意义。
- `details`：结构复杂且冗长，非必要时可先剔除（若后续需要可再引入）。

## 3. 基本清理规则

1. **去重**
   - 以 `parent_asin` 为唯一键去重。
   - 如果重复，优先保留 `title` 非空、`price` 非空的记录。

2. **缺失值处理**
   - `title` 为空的记录直接删除。
   - `main_category` 为空可填 `Unknown`，但用于评测时可过滤掉。
   - `price`, `average_rating`, `rating_number` 允许为空，后续分析时过滤。

3. **数值字段规范化**
   - `price` 转为数值（去掉 `$`、逗号等符号）。
   - `average_rating` 转为 float。
   - `rating_number` 转为 int。

4. **文本字段清洗**
   - `title`, `features`, `description`, `categories` 去除多余空白。
   - `features` / `description` 过长可截断（例如 1,000 字以内）以提升 judge 稳定性。

## 4. 输出字段（最终用于 judge）
建议输出以下字段的清洗版本：
`parent_asin`, `title`, `main_category`, `categories`, `price`, `average_rating`, `rating_number`, `store`, `features`, `description`

## 5. 示例清理脚本（无 pandas 版本）
如果本地没有 pandas，可用纯 Python 快速生成清洗版 CSV：

```python
import csv
import re

SRC = "data/industrial_and_scientific_items.csv"
DST = "data/industrial_and_scientific_items_clean.csv"

def to_float(x):
    if x is None:
        return ""
    x = x.strip()
    if not x:
        return ""
    x = re.sub(r"[^0-9.]", "", x)
    try:
        return str(float(x))
    except Exception:
        return ""

def to_int(x):
    if x is None:
        return ""
    x = x.strip()
    if not x:
        return ""
    x = re.sub(r"[^0-9]", "", x)
    try:
        return str(int(x))
    except Exception:
        return ""

keep_cols = [
    "parent_asin", "title", "main_category", "categories", "price",
    "average_rating", "rating_number", "store", "features", "description"
]

seen = {}

with open(SRC, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = (row.get("title") or "").strip()
        if not title:
            continue
        key = (row.get("parent_asin") or "").strip()
        if not key:
            continue

        row["price"] = to_float(row.get("price"))
        row["average_rating"] = to_float(row.get("average_rating"))
        row["rating_number"] = to_int(row.get("rating_number"))
        row["main_category"] = (row.get("main_category") or "Unknown").strip()

        # 去重：保留信息更完整的记录
        score = 0
        for col in ["title", "price", "average_rating", "features", "description"]:
            if (row.get(col) or "").strip():
                score += 1
        if key not in seen or score > seen[key][0]:
            seen[key] = (score, row)

with open(DST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keep_cols)
    writer.writeheader()
    for _, row in seen.values():
        out = {k: (row.get(k) or "").strip() for k in keep_cols}
        writer.writerow(out)
```

如果你确认 judge 任务需要 `details` 或 `images` 等字段，我可以再补一版带这些字段的清洗逻辑。
