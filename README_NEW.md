# CZSC Stock Analyzer v2.0 - 重构版

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/yourusername/czsc-analyzer)
[![Python](https://img.shields.io/badge/Python-3.6%2B-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

基于缠中说禅（CZSC）理论的专业股票分析工具 - 完全重构版

## 🎉 重构亮点

### ✨ 新特性
- 🏗️ **模块化架构** - 清晰的代码结构，易于维护和扩展
- 📝 **配置管理** - 统一的配置系统，支持环境变量
- 🔍 **专业日志** - 多级别日志记录，便于调试和监控
- 🧪 **可测试性** - 模块化设计使得单元测试更容易
- 🚀 **更好的性能** - 优化的连接管理和数据处理
- 📊 **命令行工具** - 强大的CLI接口，支持多种运行模式

### 🛠️ 技术改进
- ✅ 消除了代码重复（修复了两个 `connect()` 方法的问题）
- ✅ 分离关注点（数据获取、分析、监控各自独立）
- ✅ 更好的错误处理和异常管理
- ✅ 统一的代码风格和文档

## 📁 项目结构

```
czsc-analyzer/
├── app/                          # 应用程序主目录
│   ├── config/                   # 配置模块
│   │   ├── settings.py          # 配置管理
│   │   └── logging_config.py    # 日志配置
│   ├── core/                     # 核心CZSC算法
│   │   ├── data_fetcher.py      # 数据获取
│   │   ├── indicators.py        # 技术指标
│   │   ├── fractal.py           # 分型识别
│   │   ├── stroke.py            # 笔的识别
│   │   ├── pivot.py             # 中枢识别
│   │   ├── divergence.py        # 背驰检测
│   │   └── signals.py           # 买卖点识别
│   ├── services/                 # 服务层
│   │   ├── analyzer.py          # 分析服务
│   │   └── monitor.py           # 监控服务
│   ├── utils/                    # 工具模块
│   │   └── time_helper.py       # 时间工具
│   └── main.py                   # 主程序入口
├── data/                         # 数据目录
│   ├── cache/                   # 缓存
│   ├── logs/                    # 日志
│   └── db/                      # 数据库
├── tests/                        # 测试代码
├── docs/                         # 文档
├── requirements.txt              # 依赖包
├── .env.example                 # 环境变量示例
├── .gitignore                   # Git忽略文件
└── run.py                       # 快速启动脚本
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

```bash
cp .env.example .env
# 编辑 .env 文件，修改配置参数
```

### 3. 运行程序

#### 监控模式（推荐）
```bash
# 使用默认配置（5分钟刷新）
python run.py monitor

# 自定义刷新间隔（10分钟）
python run.py monitor --interval 600
```

#### 扫描模式（一次性扫描）
```bash
python run.py scan
```

#### 分析单只股票
```bash
# 分析日线
python run.py analyze --code 600000 --name 浦发银行 --market 1 --daily

# 分析30分钟线
python run.py analyze --code 600000 --name 浦发银行 --market 1 --min30

# 分析所有周期
python run.py analyze --code 600000 --name 浦发银行 --market 1 --all
```

#### 查看配置
```bash
# 显示当前配置
python run.py config --show

# 列出监控股票
python run.py config --list-stocks
```

## 📖 详细文档

### 配置管理

配置文件位于 `app/config/settings.py`，支持以下配置：

#### 通达信服务器配置
```python
servers: 服务器列表
timeout: 连接超时（秒）
retry_count: 重试次数
```

#### 监控配置
```python
interval: 监控间隔（秒）
recent_days: 日线信号范围（天）
recent_hours: 分钟线信号范围（小时）
```

#### MACD参数
```python
fast: 快速EMA周期
slow: 慢速EMA周期
signal: 信号线周期
```

### 添加监控股票

编辑 `app/config/settings.py` 中的 `StockListConfig`:

```python
stocks: List[Dict] = [
    {'code': '600000', 'name': '浦发银行', 'market': 1},  # 上海市场
    {'code': '000001', 'name': '平安银行', 'market': 0},  # 深圳市场
    # 添加更多股票...
]
```

## 🔧 开发指南

### 代码模块说明

#### 1. 数据获取层 (`core/data_fetcher.py`)
- 负责连接通达信服务器
- 获取历史和实时数据
- 自动选择最优服务器

#### 2. 技术指标层 (`core/indicators.py`)
- MACD、MA、RSI等常用指标
- KDJ、布林带等
- 可扩展其他指标

#### 3. CZSC分析层
- **分型识别** (`core/fractal.py`) - 识别顶分型和底分型
- **笔识别** (`core/stroke.py`) - 从分型形成笔
- **中枢识别** (`core/pivot.py`) - 识别价格中枢
- **背驰检测** (`core/divergence.py`) - 检测顶背驰和底背驰
- **信号识别** (`core/signals.py`) - 三类买卖点识别

#### 4. 服务层
- **分析服务** (`services/analyzer.py`) - 整合所有分析功能
- **监控服务** (`services/monitor.py`) - 自动化监控和告警

### 添加新功能

1. 在相应模块中添加新类或函数
2. 更新服务层以集成新功能
3. 添加测试用例
4. 更新文档

### 运行测试

```bash
pytest tests/
```

## 📊 CZSC理论实现

### 分型（Fractal）
识别K线图中的顶部和底部分型，作为笔的基础。

### 笔（Stroke）
连接相邻的顶分型和底分型，形成价格走势的基本单位。

### 中枢（Pivot Point）
至少三笔的重叠区间，是价格震荡的核心区域。

### 背驰（Divergence）
价格创新高/新低但指标未创新高/新低，预示趋势反转。

### 买卖点（Trading Signals）
- **第一类** - 趋势反转点（伴随背驰）
- **第二类** - 回调/反弹点
- **第三类** - 中枢突破点

## ⚙️ 系统要求

- Python 3.6+
- 稳定的网络连接（访问通达信服务器）
- 足够的内存（建议2GB+）

## 🐛 问题排查

### 连接失败
- 检查网络连接
- 尝试不同的服务器
- 查看日志文件 `data/logs/app.log`

### 数据异常
- 确认股票代码正确
- 检查市场代码（0=深圳, 1=上海）
- 查看错误日志 `data/logs/error.log`

### 性能问题
- 减少监控股票数量
- 增加监控间隔
- 使用SSD存储日志和缓存

## 📝 更新日志

### v2.0.0 (2024-01-09)
- ✨ 完全重构代码架构
- ✨ 添加配置管理系统
- ✨ 添加专业日志系统
- ✨ 优化CZSC算法实现
- ✨ 添加命令行接口
- 🐛 修复重复代码问题
- 🐛 修复日期解析问题
- 📝 完善项目文档

### v1.0.0 (Previous)
- 基础功能实现

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📜 许可证

MIT License

## 📧 联系方式

- Email: zhu_gs@126.com
- GitHub: [GaussZhu/czsc-analyzer](https://github.com/GaussZhu/czsc-analyzer)

---

**注意**: 本工具仅供学习和研究使用，不构成投资建议。股市有风险，投资需谨慎！
