# CZSC Stock Monitor - 定时任务脚本

## 📋 概述

本目录包含CZSC股票分析系统的定时任务脚本，用于自动化数据更新和分析。

---

## 📂 脚本列表

### 核心脚本

| 脚本名称 | 功能描述 | 运行频率 | 执行时间 |
|---------|---------|---------|---------|
| `fetch_daily_kline.py` | 更新K线数据（日线/周线/30分钟） | 每个交易日 | 15:25 |
| `generate_recommendations.py` | 生成买卖点推荐列表 | 每天 | 16:50 |
| `generate_tracking.py` | 生成30分钟追踪数据 | 每天 | 18:20 |

### 管理脚本

| 脚本名称 | 功能描述 |
|---------|---------|
| `install_crontab.sh` | 安装定时任务到系统crontab |
| `run_all_tasks.sh` | 手动运行所有任务（用于测试） |
| `crontab.txt` | Crontab配置文件模板 |

---

## 🚀 快速开始

### 1. 安装定时任务

```bash
cd /root/stock_monitor_project/webapp
./scripts/install_crontab.sh
```

### 2. 手动运行测试

```bash
cd /root/stock_monitor_project/webapp
./scripts/run_all_tasks.sh
```

### 3. 查看定时任务

```bash
crontab -l
```

---

## 📅 定时任务时间表

```
15:25  [每个交易日]  更新K线数据
  └─ 拉取日线、周线、30分钟K线
  └─ 存储到MySQL数据库
  
16:50  [每天]  生成推荐信号
  └─ 扫描全市场股票
  └─ 识别买点和卖点
  └─ 按市场分类
  └─ 输出 recommend_result.json
  
18:20  [每天]  生成追踪数据
  └─ 读取推荐列表中的买点
  └─ 分析30分钟K线
  └─ 计算收益指标
  └─ 输出 tracking_result.json
  
02:00  [每周一]  更新股票列表
  └─ 更新 all_stock_list.csv
  
03:00  [每天]  清理旧日志
  └─ 删除30天前的日志文件
  
*/5分钟  [持续]  监控Web服务
  └─ 检查服务是否运行
  └─ 自动重启（如果停止）
```

---

## 🔧 单独运行脚本

### 更新K线数据

```bash
cd /root/stock_monitor_project/webapp
source venv/bin/activate
python scripts/fetch_daily_kline.py
```

**功能**:
- 从TDX服务器拉取K线数据
- 支持日线、周线、30分钟三种周期
- 增量更新（只插入新数据）
- 自动选择最优TDX服务器

**日志**: `logs/fetch_kline_YYYYMMDD.log`

---

### 生成推荐信号

```bash
cd /root/stock_monitor_project/webapp
source venv/bin/activate
python scripts/generate_recommendations.py
```

**功能**:
- 扫描全市场股票
- 识别未闭环买点和孤立卖点
- 按市场分类（上证/深证/创业板/科创板/北交所）
- 每个市场取前5个推荐

**输出**: `web/static/recommend_result.json`

**日志**: `logs/recommend_YYYYMMDD.log`

---

### 生成追踪数据

```bash
cd /root/stock_monitor_project/webapp
source venv/bin/activate
python scripts/generate_tracking.py
```

**功能**:
- 读取推荐列表中的买点
- 获取30分钟K线数据
- 计算当前收益、最大收益、最大回撤
- 检查是否有卖点（闭环）

**输出**: `web/static/tracking_result.json`

**日志**: `logs/tracking_YYYYMMDD.log`

---

## 📊 数据流程图

```
                    ┌─────────────────┐
                    │  股票列表 CSV   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ TDX服务器       │
                    │ (实时K线数据)   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  日线数据      │  │  周线数据      │  │  30分钟数据    │
│  (600条)       │  │  (240条)       │  │  (800条)       │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  MySQL数据库    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼─────────┐ ┌───────▼─────────┐ ┌───────▼─────────┐
│ 分析引擎(CZSC)  │ │ 推荐信号生成    │ │ 30分钟追踪      │
│ - 分型识别      │ │ - 买点推荐      │ │ - 收益计算      │
│ - 笔段划分      │ │ - 卖点推荐      │ │ - 风险指标      │
│ - 中枢判断      │ │ - 市场分类      │ │ - 闭环判断      │
│ - 买卖点识别    │ └─────────────────┘ └─────────────────┘
└─────────────────┘          │                    │
                             │                    │
                    ┌────────▼────────┐  ┌────────▼────────┐
                    │ recommend.json  │  │ tracking.json   │
                    └────────┬────────┘  └────────┬────────┘
                             │                    │
                             └────────┬───────────┘
                                      │
                             ┌────────▼────────┐
                             │  Web前端展示    │
                             │  - 推荐列表     │
                             │  - 追踪列表     │
                             └─────────────────┘
```

---

## 📝 日志管理

### 日志位置

```
logs/
├── fetch_kline_YYYYMMDD.log      # K线更新日志
├── recommend_YYYYMMDD.log         # 推荐信号日志
├── tracking_YYYYMMDD.log          # 追踪数据日志
├── update_list_YYYYMMDD.log       # 股票列表更新日志
├── web_service.log                # Web服务日志
└── manual_*.log                   # 手动运行日志
```

### 查看日志

```bash
# 查看今天的K线更新日志
tail -f logs/fetch_kline_$(date +%Y%m%d).log

# 查看推荐信号日志
tail -f logs/recommend_$(date +%Y%m%d).log

# 查看追踪数据日志
tail -f logs/tracking_$(date +%Y%m%d).log

# 查看所有日志
ls -lh logs/
```

### 日志清理

定时任务会自动清理30天前的日志：

```bash
# 手动清理
find logs/ -name "*.log" -type f -mtime +30 -delete
```

---

## 🔍 故障排查

### 问题1: 定时任务不执行

**检查步骤**:
1. 确认crontab已安装: `crontab -l`
2. 检查cron服务状态: `systemctl status cron`
3. 查看系统日志: `grep CRON /var/log/syslog`

**解决方案**:
```bash
# 重启cron服务
sudo systemctl restart cron

# 重新安装定时任务
./scripts/install_crontab.sh
```

---

### 问题2: 脚本执行失败

**检查步骤**:
1. 查看错误日志: `tail -100 logs/fetch_kline_$(date +%Y%m%d).log`
2. 手动运行测试: `./scripts/run_all_tasks.sh`
3. 检查Python环境: `venv/bin/python --version`

**常见原因**:
- 虚拟环境未激活
- 依赖包未安装
- 数据库连接失败
- TDX服务器不可用

---

### 问题3: 数据未更新

**检查步骤**:
1. 确认脚本执行成功（查看日志）
2. 检查输出文件: `ls -lh web/static/*.json`
3. 验证数据格式: `python -m json.tool web/static/recommend_result.json`

**解决方案**:
```bash
# 手动重新生成
cd /root/stock_monitor_project/webapp
source venv/bin/activate
python scripts/generate_recommendations.py
python scripts/generate_tracking.py
```

---

## ⚙️ 配置说明

### 修改执行时间

编辑 `crontab.txt`：

```bash
# 修改K线更新时间为15:30
30 15 * * 1-5 cd $WEBAPP_ROOT && ...
```

重新安装：

```bash
./scripts/install_crontab.sh
```

---

### 修改数据数量

编辑脚本中的参数：

```python
# fetch_daily_kline.py
periods = [
    {"period": "daily", "count": 600},   # 600 → 1200
    {"period": "weekly", "count": 240},  # 240 → 480
    {"period": "30min", "count": 800},   # 800 → 1600
]
```

---

### 自定义推荐数量

编辑 `generate_recommendations.py`：

```python
# 每个市场取前5个改为前10个
)[:5]  # 改为 )[:10]
```

---

## 📊 性能指标

### 执行时间（参考）

| 任务 | 股票数量 | 执行时间 |
|-----|---------|---------|
| K线更新 | 5000只 | ~30分钟 |
| 推荐生成 | 5000只 | ~45分钟 |
| 追踪数据 | 25只 | ~2分钟 |

### 资源消耗

- **CPU**: 单核10-30%
- **内存**: 500MB-1GB
- **磁盘**: 每天增长约100MB
- **网络**: K线更新时约50MB/min

---

## 🔐 安全建议

1. **日志权限**: 确保日志目录权限正确
   ```bash
   chmod 755 logs/
   chmod 644 logs/*.log
   ```

2. **数据备份**: 定期备份JSON文件
   ```bash
   cp web/static/*.json backups/
   ```

3. **错误通知**: 配置邮件提醒（可选）
   ```bash
   # 在crontab中添加MAILTO
   MAILTO=your_email@example.com
   ```

---

## 📚 相关文档

- [系统部署指南](../DEPLOY_AND_PR_GUIDE.md)
- [功能完成报告](../ALL_FEATURES_COMPLETE.md)
- [API文档](../docs/API.md)

---

## 🆘 获取帮助

如遇问题，请：
1. 查看日志文件
2. 运行手动测试脚本
3. 检查GitHub Issues
4. 联系开发团队

---

**更新时间**: 2026-01-10  
**版本**: v2.2
