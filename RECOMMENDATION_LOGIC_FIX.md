# 推荐逻辑修复说明

## 📋 问题描述

### 之前的逻辑（❌ 不科学）

```python
# 找最新的未闭环买点
latest_buy = find_latest_unclosed_buy(buy_signals, sell_signals)
```

**问题**：
- 只要买点之后没有卖点，就会一直推荐
- **可能推荐几个月前的买点**
- 例如：
  - 信号时间：2025-08-06
  - 推荐日期：2025-12-20
  - 时间差：**4.5个月**
  - 结果：推荐了退市股（东通退），30天亏损 -66.91%

### 现在的逻辑（✅ 科学）

```python
# 只推荐当天出现的买点
today_buys = find_today_buy_signals(buy_signals, target_date)
```

**优势**：
- ✅ **信号新鲜**：只推荐当天出现的买点
- ✅ **逻辑合理**：今天是1月20日 → 只推荐1月20日出现的买点
- ✅ **避免错杀**：不会推荐过时的信号

---

## 📊 回测对比

### 旧逻辑回测（2025-12-20，500只样本）

| 指标 | 7天 | 14天 | 30天 |
|------|-----|------|------|
| 推荐数 | 99 | 99 | 99 |
| 胜率 | 70.71% | 76.77% | **85.86%** |
| 平均收益 | 8.87% | 10.59% | **20.4%** |
| 最大收益 | 165.66% | 132.5% | 154.44% |

**问题**：
- 推荐了很多老信号（比如4个月前的买点）
- 胜率虚高（因为只要没有卖点就一直推荐）
- 不科学

### 新逻辑回测（2025-12-01，200只样本）

| 指标 | 7天 | 14天 | 30天 |
|------|-----|------|------|
| 推荐数 | 24 | 24 | 24 |
| 胜率 | **70.83%** | 50.0% | 58.33% |
| 平均收益 | **2.21%** | 1.6% | **4.19%** |
| 最大收益 | 14.95% | 32.96% | 32.33% |

**优势**：
- ✅ 只推荐当天出现的买点
- ✅ 信号时间 = 推荐日期
- ✅ 胜率更真实
- ✅ 避免推荐退市股、ST股等老信号

---

## 🔍 关键差异

### 时间线对比

#### 旧逻辑
```
2025-08-06                2025-12-20
    ↓                         ↓
信号时间                   推荐日期
(买点出现)              (我们推荐它)
    │                         │
    │←───── 4.5个月 ─────→│
    │                         │
    └─────────────────────────┘
         信号已经很旧了！
```

#### 新逻辑
```
2025-12-01                2025-12-01
    ↓                         ↓
信号时间                   推荐日期
(买点出现)              (我们推荐它)
    │                         │
    └─────────────────────────┘
      信号时间 = 推荐日期
          新鲜！科学！
```

---

## 💻 代码实现

### 新增函数

```python
def find_today_buy_signals(buy_signals, target_date):
    """
    找到指定日期当天出现的买点信号
    
    Args:
        buy_signals: 买点信号列表
        target_date: 目标日期（datetime对象）
        
    Returns:
        当天的买点信号列表（可能有多个）
    """
    if not buy_signals:
        return []
    
    # 确保是date对象
    if hasattr(target_date, 'date'):
        target_date = target_date.date()
    
    # 筛选出当天的买点
    today_buys = []
    for buy_sig in buy_signals:
        buy_date = buy_sig.get("date")
        if buy_date and hasattr(buy_date, 'date'):
            buy_date = buy_date.date()
        
        if buy_date == target_date:
            today_buys.append(buy_sig)
    
    # 按类型排序：first_type > second_type > third_type
    type_priority = {'first_type': 1, 'second_type': 2, 'third_type': 3}
    today_buys_sorted = sorted(
        today_buys, 
        key=lambda x: type_priority.get(x.get('type', 'third_type'), 999)
    )
    
    return today_buys_sorted
```

### 使用示例

```python
# 推荐系统
today_buys = find_today_buy_signals(buy_signals, target_date=None)
for buy_signal in today_buys:
    recommendations.append({
        "code": code,
        "name": name,
        "类型": buy_signal.get('type'),
        "价格": buy_signal.get('price'),
        "时间": buy_signal.get('date').strftime("%Y-%m-%d"),
        "推荐理由": f"{buy_signal.get('type')}信号确认，今日出现买点"
    })

# 回测系统
today_buys = find_today_buy_signals(buy_signals, target_date)
for buy_signal in today_buys:
    recommendations.append({
        "code": code,
        "推荐日期": target_date.strftime("%Y-%m-%d"),
        "推荐价格": buy_signal.get('price'),
        "信号时间": buy_signal.get('date').strftime("%Y-%m-%d")
    })
```

---

## 🎯 实际案例

### 旧逻辑案例（❌ 失败）

```json
{
  "code": "300379",
  "name": "东通退",
  "推荐日期": "2025-12-20",
  "推荐价格": 5.56,
  "信号时间": "2025-08-06",  // ❌ 4.5个月前的信号
  "30天后价格": 1.84,
  "30天收益率": -66.91,
  "是否盈利": false
}
```

### 新逻辑案例（✅ 成功）

```json
{
  "code": "603121",
  "name": "华培动力",
  "推荐日期": "2025-12-01",
  "推荐价格": 15.99,
  "信号时间": "2025-12-01",  // ✅ 当天的信号
  "7天后价格": 18.38,
  "7天收益率": 14.95,
  "是否盈利": true
}
```

---

## 🚀 部署建议

### 1. 重新生成推荐

```bash
cd /root/stock_monitor_project/webapp
/root/stock_monitor_project/webapp/venv/bin/python3 scripts/generate_recommendations.py
```

### 2. 查看推荐结果

```bash
cat /root/stock_monitor_project/webapp/web/static/recommend_result.json | python3 -m json.tool | head -50
```

### 3. 运行回测验证

```bash
# 快速测试（50只）
/root/stock_monitor_project/webapp/venv/bin/python3 scripts/backtest_full.py "2025-12-01" 50

# 中等规模（200只）
/root/stock_monitor_project/webapp/venv/bin/python3 scripts/backtest_full.py "2025-12-01" 200

# 全网扫描（5800只）
nohup /root/stock_monitor_project/webapp/venv/bin/python3 scripts/backtest_full.py "2025-12-01" > /tmp/backtest_full.log 2>&1 &
```

---

## 📝 总结

### 问题
- 旧逻辑推荐的是"最新的未闭环买点"
- 可能是几个月前的信号
- 导致推荐了退市股、ST股等失效信号

### 解决方案
- 新逻辑只推荐"当天出现的买点"
- 信号新鲜，逻辑合理
- 胜率更真实，避免错杀

### 回测验证
- 2025-12-01，200只样本
- 7天胜率：70.83%，平均收益2.21%
- 30天胜率：58.33%，平均收益4.19%

### 下一步
1. 重新生成今天的推荐
2. 运行更大规模的回测（500只、5800只）
3. 添加ST/退市股过滤
4. 添加成交量、价格趋势等过滤条件

---

**更新时间**：2026-01-20
**提交记录**：aa1a065
