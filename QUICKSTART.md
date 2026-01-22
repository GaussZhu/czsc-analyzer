# 🚀 区间套系统快速开始

## 5分钟快速上手

### 1️⃣ 环境准备

```bash
cd /root/stock_monitor_project/webapp
source venv/bin/activate
```

### 2️⃣ 测试示例

```bash
python3 test_interval_nesting.py
```

**预期输出**：
```
✅ 600000 浦发银行 - 置信度: 95% 🔥
✅ 000001 平安银行 - 置信度: 90% 🔥
```

---

## 📚 快速示例

### 分析单只股票

```python
from app.services.interval_nesting import (
    IntervalNestingAnalyzer, 
    format_signal_report
)

# 初始化
analyzer = IntervalNestingAnalyzer(use_mysql=True)

# 分析
signal = analyzer.analyze(
    code='600000',
    name='浦发银行',
    market=1,
    check_intraday=True
)

# 查看结果
if signal:
    print(format_signal_report(signal))
    
    # 关键信息
    print(f"置信度: {signal.confidence:.0%}")
    print(f"入场: ¥{signal.entry_zone.low:.2f} - ¥{signal.entry_zone.high:.2f}")
    print(f"止损: ¥{signal.stop_loss:.2f}")
    print(f"目标: ¥{signal.target_price:.2f}")
    print(f"盈亏比: {signal.risk_reward:.2f}:1")

analyzer.close()
```

### 批量扫描

```python
# 准备股票列表
stocks = [
    ('600000', '浦发银行', 1),
    ('000001', '平安银行', 0),
    ('600036', '招商银行', 1),
    # ... 添加更多
]

# 扫描（只返回置信度≥70%的信号）
signals = analyzer.scan_market(
    stock_list=stocks,
    min_confidence=0.7
)

# 查看前10个
for signal in signals[:10]:
    print(f"{signal.code} {signal.name}")
    print(f"  置信度: {signal.confidence:.0%}")
    print(f"  {signal.summary}")
    print(f"  入场: ¥{signal.entry_zone.optimal:.2f}")
    print()
```

---

## 🎯 核心逻辑

### 三层分析

```
周线 → 确认大趋势（上涨/震荡/下跌）
  ↓
日线 → 寻找买点（一买/二买/三买）
  ↓
30分钟 → 精确入场点
```

### 自动过滤

```python
❌ 周线下跌 → 直接过滤
❌ 日线无买点 → 返回None
✅ 其他情况 → 生成信号
```

---

## 📊 评级标准

| 置信度 | 评级 | 说明 |
|--------|------|------|
| 85%+ 🔥 | 强烈推荐 | 大级别上涨+日线二买+小级别共振 |
| 75%+ ✓ | 推荐 | 日线二买或三买+周线配合 |
| 65%+ ⚠ | 可关注 | 日线有买点，但缺乏共振 |
| 65%- ❌ | 谨慎 | 仅一买或无小级别支持 |

---

## 💰 操作建议

### 入场策略

1. **分批建仓**
   - 第一批：区间下沿（如 ¥11.13）
   - 第二批：最优价格（如 ¥11.36）
   - 第三批：区间上沿（如 ¥11.70）

2. **仓位管理**
   - 85%+ 置信度：30-40% 仓位
   - 75%+ 置信度：20-30% 仓位
   - 65%+ 置信度：10-20% 仓位

### 止损策略

- **一买**：-8%（风险大）
- **二买**：-5%（风险中）
- **三买**：-5%（风险小）

### 目标价策略

- **一买**：+20%（空间大）
- **二买**：+15%（稳健）
- **三买**：+12%（保守）

---

## 📖 完整文档

- **使用指南**：[INTERVAL_NESTING_GUIDE.md](INTERVAL_NESTING_GUIDE.md)
- **实现总结**：[INTERVAL_NESTING_SUMMARY.md](INTERVAL_NESTING_SUMMARY.md)
- **完成报告**：[TASK_COMPLETION_REPORT.md](TASK_COMPLETION_REPORT.md)

---

## 🔧 常见问题

### Q1: 系统不支持5分钟级别？

**A**: 目前5分钟实时获取功能待实现，但不影响核心功能。系统主要依赖：
- 周线：确认趋势
- 日线：找买点（核心）
- 30分钟：精确入场

### Q2: 置信度是如何计算的？

**A**: 置信度由多个因素综合：
```
基础分 50%
+ 日线买点（二买+30%，三买+20%，一买+10%）
+ 周线趋势（上涨+20%，震荡+10%）
+ 30分钟共振（+10%）
= 最终置信度（最高95%）
```

### Q3: 如何选择高质量信号？

**A**: 优先选择：
1. 置信度 ≥ 75%
2. 日线二买或三买
3. 有30分钟共振
4. 盈亏比 ≥ 2:1

### Q4: 数据从哪里来？

**A**: 
- 周线/日线/30分钟：MySQL数据库
- 5分钟：实时从TDX获取（待实现）

---

## ⚠️ 风险提示

### 系统局限

1. **历史数据延迟**：日线可能有1-2天延迟
2. **市场环境影响**：单边下跌市效果差
3. **小级别噪音**：5分钟/1分钟容易假信号

### 使用建议

- ✅ 只操作高置信度信号（≥75%）
- ✅ 严格止损，不心存侥幸
- ✅ 分批建仓，不一次性全仓
- ✅ 结合基本面，避开业绩地雷
- ✅ 关注市场情绪，大盘暴跌时观望

---

## 📞 获取帮助

- **查看代码**：`app/services/interval_nesting.py`
- **运行测试**：`python3 test_interval_nesting.py`
- **查看文档**：`INTERVAL_NESTING_GUIDE.md`

---

**最后更新**：2026-01-17  
**版本**：v2.0  
**状态**：✅ 生产就绪
