# 实时追踪系统说明文档

## 📋 系统概述

实时追踪系统是一个完整的股票监控解决方案，能够自动追踪推荐股票的实时表现，并提供丰富的数据展示和分析功能。

## 🎯 核心功能

### 1. 实时行情更新
- **数据源**：腾讯财经API（免费、无限制）
- **更新频率**：交易时间内每5分钟更新一次
- **更新时段**：
  - 上午：9:30-11:30（每5分钟）
  - 下午：13:00-15:00（每5分钟）
- **非交易时间**：自动跳过更新，节省资源

### 2. 数据展示
#### 持仓概览卡片
- 📊 总追踪数
- ✅ 盈利股票数量及占比
- ❌ 亏损股票数量及占比
- 💰 平均收益率

#### 历史胜率统计
- 7天胜率（可视化进度条）
- 14天胜率（可视化进度条）
- 30天胜率（可视化进度条）

#### Top/Bottom榜单
- 🏆 收益Top5：表现最好的5只股票
- 📉 亏损Top5：表现最差的5只股票

#### 风险提示
- ⚠️ 跌幅超过5%的股票预警
- ⏰ 持有超过30天的股票提醒

#### 智能过滤器
- 全部股票
- 仅盈利股票
- 仅亏损股票
- 按买点类型筛选（1买/2买/3买）

#### 详细股票列表
- 股票代码和名称（可点击查看详情）
- 推荐类型（first_type/second_type/third_type）
- 推荐日期
- 价格变化（推荐价 → 当前价）
- 收益率（红涨绿跌）
- 持有天数
- 快捷操作按钮

### 3. 前端自动刷新
- **刷新频率**：每5分钟自动重新加载数据
- **最后更新时间**：实时显示最后刷新时间
- **无需手动刷新**：页面会自动更新最新数据

## 🔧 技术架构

### 后端脚本

#### 1. `update_realtime_prices.py` - 实时行情更新
```python
功能：使用腾讯财经API获取股票实时价格
输入：tracking_history.json（追踪股票列表）
输出：更新后的tracking_history.json（带最新价格）
运行时间：交易日9:30-15:00，每5分钟
```

#### 2. `daily_tracking.py` - 每日跟踪更新
```python
功能：每天更新追踪股票列表，添加新推荐股票
输入：recommend_result.json（每日推荐）
输出：
  - tracking_history.json（更新的追踪列表）
  - win_rate_stats.json（胜率统计）
运行时间：每天19:00
```

#### 3. `generate_recommendations.py` - 生成推荐
```python
功能：扫描全市场，生成当天的买点推荐
输入：all_stock_list.csv（股票列表）
输出：recommend_result.json（当天推荐）
运行时间：每天16:20（收盘后）
```

### 前端页面

#### `/tracking` - 实时追踪页面
```javascript
数据源：
  - /static/tracking_history.json（追踪数据）
  - /static/win_rate_stats.json（胜率统计）

刷新策略：
  - 初始加载：页面打开时立即加载
  - 自动刷新：每5分钟重新加载数据
  - 手动刷新：用户刷新页面时重新加载

响应式设计：
  - 桌面端：4列卡片布局
  - 移动端：单列布局
```

## ⏰ 定时任务配置

系统通过crontab配置了多个定时任务：

```bash
# 1. 更新股票列表（每个交易日9:25）
25 9 * * 1-5 cd /root/stock_monitor_project && \
  /root/stock_monitor_project/venv/bin/python scripts/update_stock_list.py

# 2. K线数据更新（每个交易日15:25）
25 15 * * 1-5 cd /root/stock_monitor_project && \
  /root/stock_monitor_project/venv/bin/python3 -m scripts.fetch_daily_kline --mode incremental

# 3. 生成推荐信号（每天16:20）
20 16 * * * cd /root/stock_monitor_project/webapp && \
  /root/stock_monitor_project/webapp/venv/bin/python3 scripts/generate_recommendations.py

# 4. 更新每日跟踪（每天19:00）
0 19 * * * cd /root/stock_monitor_project/webapp && \
  /root/stock_monitor_project/webapp/venv/bin/python3 scripts/daily_tracking.py

# 5. 实时行情更新（交易日9:30-15:00，每5分钟）
*/5 9-11 * * 1-5 cd /root/stock_monitor_project/webapp && \
  /root/stock_monitor_project/webapp/venv/bin/python3 scripts/update_realtime_prices.py

*/5 13-14 * * 1-5 cd /root/stock_monitor_project/webapp && \
  /root/stock_monitor_project/webapp/venv/bin/python3 scripts/update_realtime_prices.py
```

## 📊 数据流程

### 每日工作流程

```
09:25 → 更新股票列表
        ↓
09:30-15:00 → 实时行情更新（每5分钟）
        ↓
15:25 → K线数据更新
        ↓
16:20 → 生成当天推荐
        ↓
19:00 → 更新追踪列表
        ↓
持续运行 → 前端自动刷新（每5分钟）
```

### 数据文件说明

#### `tracking_history.json`
```json
[
  {
    "code": "300587",
    "name": "天铁科技",
    "market": 1,
    "推荐日期": "2026-01-19",
    "推荐价格": 6.35,
    "当前价格": 6.50,
    "涨跌幅": 2.36,
    "持有天数": 1,
    "推荐类型": "third_type",
    "推荐理由": "third_type信号确认，等待放量突破",
    "市场": "创业板",
    "更新时间": "2026-01-22 10:15:33"
  }
]
```

#### `win_rate_stats.json`
```json
{
  "7天": {
    "总数": 24,
    "盈利": 17,
    "亏损": 7,
    "胜率": 70.83,
    "平均收益": 2.21,
    "最大收益": 14.95,
    "最大亏损": -4.75
  },
  "14天": {...},
  "30天": {...}
}
```

## 🚀 使用指南

### 手动运行脚本

#### 1. 更新实时行情
```bash
cd /root/stock_monitor_project/webapp
/root/stock_monitor_project/webapp/venv/bin/python3 scripts/update_realtime_prices.py
```

#### 2. 更新追踪列表
```bash
cd /root/stock_monitor_project/webapp
/root/stock_monitor_project/webapp/venv/bin/python3 scripts/daily_tracking.py
```

#### 3. 生成推荐
```bash
cd /root/stock_monitor_project/webapp
/root/stock_monitor_project/webapp/venv/bin/python3 scripts/generate_recommendations.py
```

### 查看日志

```bash
# 实时行情日志
tail -f /root/stock_monitor_project/logs/realtime_prices_$(date +%F).log

# 追踪更新日志
tail -f /root/stock_monitor_project/logs/tracking_$(date +%F).log

# 推荐生成日志
tail -f /root/stock_monitor_project/logs/recommend_$(date +%F).log
```

### 访问前端页面

```
http://localhost:8000/tracking
```

## 🔍 API说明

### 免费行情API

#### 腾讯财经API（推荐）
```
URL: http://qt.gtimg.cn/q={symbols}
示例: http://qt.gtimg.cn/q=sz000001,sh600000
返回格式: GBK编码的文本数据

字段说明（以~分隔）：
[0] 未知
[1] 名称
[2] 代码
[3] 当前价格 ⭐
[4] 昨收
[5] 今开
[6] 成交量（手）
[7] 外盘
[8] 内盘
[9] 买一
[10] 买一量（手）
... 更多字段
```

#### 东方财富API（备用）
```
URL: http://push2.eastmoney.com/api/qt/stock/get
参数:
  - secid: 市场代码.股票代码（如 0.000001）
  - fields: f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58

返回格式: JSON
关键字段:
  - f43: 当前价格 ⭐
  - f44: 最高价
  - f45: 最低价
  - f46: 今开
  - f47: 成交量
  - f48: 成交额
```

## 🎨 UI/UX设计

### 配色方案
- **主色调**：红色（#b22222） - 代表热情和活力
- **涨跌配色**：
  - 盈利：绿色（#28a745）
  - 亏损：红色（#dc3545）
- **背景**：渐变白色到浅粉色
- **卡片**：白色背景，带阴影和悬停效果

### 响应式布局
- **桌面端**（>768px）：
  - 概览卡片：4列网格
  - Top/Bottom榜单：2列并排
  - 胜率统计：3列网格
- **移动端**（≤768px）：
  - 所有内容：单列布局
  - 自适应字体大小

### 交互设计
- **卡片悬停**：上浮效果 + 阴影加深
- **按钮悬停**：边框变色 + 文字变色
- **加载状态**：动画省略号
- **数据更新**：平滑过渡效果

## 📈 性能优化

### 1. API调用优化
- 批量查询：每次最多50只股票
- 请求间隔：0.5秒（避免请求过快）
- 异常处理：自动跳过失败的请求
- 交易时间判断：非交易时间不请求

### 2. 前端优化
- 数据缓存：减少重复加载
- 懒加载：按需渲染列表
- 节流/防抖：避免频繁操作
- 本地计算：客户端过滤和排序

### 3. 文件存储
- JSON格式：易读易写
- 增量更新：只更新变化的数据
- 文件压缩：减少体积（可选）

## 🔒 稳定性保障

### 1. 错误处理
```python
try:
    # 获取行情数据
    response = requests.get(url, timeout=10)
except Exception as e:
    logger.error(f"获取行情失败: {e}")
    continue  # 跳过失败的股票，继续处理其他股票
```

### 2. 日志记录
- 详细的运行日志
- 错误和异常追踪
- 性能统计信息

### 3. 数据备份
- 定期备份tracking_history.json
- 错误时保留旧数据
- 版本控制管理

## 🎯 未来优化方向

### 1. 功能扩展
- [ ] 止损止盈设置
- [ ] 导出Excel/PDF报告
- [ ] 微信/邮件通知
- [ ] 更多图表展示（K线图、收益曲线）
- [ ] 自定义跟踪列表

### 2. 性能提升
- [ ] WebSocket实时推送
- [ ] Redis缓存
- [ ] 数据库存储
- [ ] 并发请求优化

### 3. 数据增强
- [ ] 更多行情数据源
- [ ] 历史数据回溯
- [ ] 更多技术指标
- [ ] AI预测辅助

## 📞 问题反馈

如有问题或建议，请联系开发团队。

---

**最后更新时间**: 2026-01-22
**版本号**: v1.0.0
**维护者**: Stock Monitor Team
