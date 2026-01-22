# MySQL数据库集成指南

## 📚 概述

CZSC股票分析工具现已完全集成MySQL数据库支持，可以：
- ✅ 从MySQL读取历史K线数据
- ✅ 自动切换TDX和MySQL数据源
- ✅ 支持多种周期（日线、30分钟、周线）
- ✅ 访问Kronos预测数据

---

## 🗄️ 数据库结构

你的 `stock_kline` 数据库包含以下表：

| 表名 | 用途 | 数据周期 |
|------|------|----------|
| `daily_kline` | 日线K线数据 | 日 |
| `30min_kline` | 30分钟K线数据 | 30分钟 |
| `weekly_kline` | 周线K线数据 | 周 |
| `code_map` | 股票代码映射 | - |
| `bj_code_map` | 北交所股票代码 | - |
| `kronos_pred` | Kronos预测数据 | - |
| `kronos_eval_daily` | Kronos每日评估 | 日 |

---

## ⚙️ 配置

### 1. 环境变量配置

创建 `.env` 文件（或修改现有的）：

```bash
# MySQL Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=stock_kline
USE_MYSQL_DATA=True
```

### 2. 配置说明

- `MYSQL_HOST`: MySQL服务器地址
- `MYSQL_PORT`: MySQL端口（默认3306）
- `MYSQL_USER`: 数据库用户名
- `MYSQL_PASSWORD`: 数据库密码
- `MYSQL_DATABASE`: 数据库名称（你的是 `stock_kline`）
- `USE_MYSQL_DATA`: 是否优先使用MySQL数据（`True`/`False`）

---

## 🚀 使用方法

### 方法1: 自动模式（推荐）

系统会自动选择最佳数据源：
1. 优先尝试MySQL
2. 如果MySQL失败或未配置，自动切换到TDX

```bash
# 无需修改任何代码，直接运行
python run.py monitor
```

### 方法2: 强制使用MySQL

```python
from app.services.analyzer import StockAnalyzer

# 强制使用MySQL数据源
analyzer = StockAnalyzer(use_mysql=True)
result = analyzer.analyze_stock('600000', '浦发银行', 1, 'daily')
```

### 方法3: 强制使用TDX

```python
from app.services.analyzer import StockAnalyzer

# 强制使用TDX实时数据
analyzer = StockAnalyzer(use_mysql=False)
result = analyzer.analyze_stock('600000', '浦发银行', 1, 'daily')
```

---

## 🧪 测试MySQL连接

运行测试脚本验证数据库连接：

```bash
cd /root/stock_monitor_project/webapp
python scripts/test_mysql_connection.py
```

测试脚本会：
- ✅ 测试数据库连接
- ✅ 显示所有表
- ✅ 获取股票列表
- ✅ 读取K线数据示例
- ✅ 显示数据范围统计

---

## 📊 数据访问API

### 1. 使用StockDAO直接访问

```python
from app.core.db_connector import MySQLConnector
from app.core.stock_dao import StockDAO

# 创建连接
connector = MySQLConnector(
    host='localhost',
    user='root',
    password='your_password',
    database='stock_kline'
)

# 创建DAO
dao = StockDAO(connector)

# 获取日线数据
daily_data = dao.get_daily_kline('600000', limit=100)

# 获取30分钟数据
min30_data = dao.get_30min_kline('600000', limit=200)

# 获取周线数据
weekly_data = dao.get_weekly_kline('600000', limit=50)

# 获取股票列表
stocks = dao.get_all_stock_codes()

# 获取数据范围
data_range = dao.get_data_range('daily_kline', '600000')
```

### 2. 使用统一数据源

```python
from app.core.data_source import UnifiedDataSource, DataSourceType

# 自动选择数据源
data_source = UnifiedDataSource(DataSourceType.AUTO)

# 获取数据（优先MySQL）
data = data_source.get_stock_data(
    stock_code='600000',
    market=1,
    period='daily',
    count=300,
    prefer_mysql=True
)

# 从数据库获取股票列表
stock_list = data_source.get_stock_list_from_db()
```

---

## 🔧 进阶功能

### 自定义SQL查询

```python
from app.core.db_connector import MySQLConnector

connector = MySQLConnector(...)

# 执行自定义查询
query = """
    SELECT code, name, COUNT(*) as count 
    FROM daily_kline d
    JOIN code_map c ON d.code = c.code
    WHERE d.date >= '2024-01-01'
    GROUP BY code, name
"""
results = connector.execute_query(query)

# 转换为DataFrame
df = connector.query_to_dataframe(query)
```

### 批量插入数据

```python
# 批量插入分析结果
data = [
    {'code': '600000', 'date': '2024-01-09', 'signal': 'buy'},
    {'code': '600004', 'date': '2024-01-09', 'signal': 'sell'},
]

connector.batch_insert('analysis_results', data, ignore=True)
```

---

## 📈 数据格式说明

### K线数据字段

MySQL表中的数据会自动转换为标准格式：

| MySQL字段 | 标准字段 | 说明 |
|-----------|----------|------|
| `date` 或 `datetime` | `Date` (索引) | 日期/时间 |
| `open` | `Open` | 开盘价 |
| `high` | `High` | 最高价 |
| `low` | `Low` | 最低价 |
| `close` | `Close` | 收盘价 |
| `volume` 或 `vol` | `Volume` | 成交量 |
| `amount` | `Amount` | 成交额 |

---

## ⚡ 性能优化

### 1. 使用MySQL的优势
- ✅ 历史数据读取更快
- ✅ 不受TDX服务器限制
- ✅ 可以批量分析多只股票
- ✅ 支持复杂查询和统计

### 2. 数据源选择策略

```python
# 实时分析 - 使用TDX获取最新数据
analyzer = StockAnalyzer(use_mysql=False)

# 历史回测 - 使用MySQL速度更快
analyzer = StockAnalyzer(use_mysql=True)

# 日常监控 - 自动模式（优先MySQL）
analyzer = StockAnalyzer(use_mysql=None)  # 或不传参数
```

### 3. 缓存策略

数据自动缓存在内存中，避免重复查询。

---

## 🔍 故障排查

### 问题1: 连接失败

```
✗ MySQL连接失败: (2003, "Can't connect to MySQL server")
```

**解决方法**:
- 检查MySQL服务是否运行
- 确认host和port配置正确
- 检查防火墙设置

### 问题2: 认证失败

```
✗ MySQL连接失败: (1045, "Access denied for user...")
```

**解决方法**:
- 确认用户名和密码正确
- 确认用户有访问数据库的权限

### 问题3: 数据库不存在

```
✗ MySQL连接失败: (1049, "Unknown database 'stock_kline'")
```

**解决方法**:
- 确认数据库名称正确
- 检查数据库是否已创建

### 问题4: 表不存在

```
⚠ 表不存在: daily_kline
```

**解决方法**:
- 确认表名大小写匹配
- 检查表是否已创建

---

## 📝 最佳实践

### 1. 配置管理

```bash
# 开发环境 - 使用本地MySQL
MYSQL_HOST=localhost
USE_MYSQL_DATA=True

# 生产环境 - 使用TDX实时数据
USE_MYSQL_DATA=False
```

### 2. 数据同步

定期从TDX更新MySQL数据，保持数据最新：

```python
# TODO: 创建数据同步脚本
# scripts/sync_tdx_to_mysql.py
```

### 3. 备份策略

定期备份MySQL数据库：

```bash
mysqldump -u root -p stock_kline > backup_$(date +%Y%m%d).sql
```

---

## 🔮 未来扩展

### 计划功能

- [ ] **分析结果保存** - 将CZSC分析结果保存到数据库
- [ ] **回测功能** - 基于历史数据进行策略回测
- [ ] **数据同步** - 自动从TDX同步最新数据到MySQL
- [ ] **性能优化** - 添加Redis缓存层
- [ ] **多数据库支持** - 支持PostgreSQL等其他数据库

### 创建分析结果表

```sql
CREATE TABLE czsc_analysis_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    period VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50),
    signal_price DECIMAL(10,2),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_code_date (code, date),
    INDEX idx_signal_type (signal_type)
);
```

---

## 📞 技术支持

如果遇到问题：
1. 查看日志文件：`data/logs/app.log`
2. 运行测试脚本诊断
3. 检查配置文件

---

**版本**: v2.0
**更新时间**: 2024-01-09
