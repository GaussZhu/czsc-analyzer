# Web 应用重构计划

## 📋 当前状态分析

### 原始 Web 应用结构

```
stock_monitor_project/
├── app.py                          # FastAPI 主应用 (682行)
├── api/
│   ├── routes_api.py               # API 路由 (26KB)
│   └── models.py                   # 数据模型
├── templates/
│   ├── index.html                  # 首页
│   └── analysis_result.html        # 分析结果页
├── static/
│   ├── css/                        # 样式文件
│   ├── js/                         # JavaScript 文件
│   └── data/                       # 静态数据
├── run_server.sh                   # 启动脚本
├── stop_server.sh                  # 停止脚本
└── user_routes.py                  # 用户路由
```

### 域名部署

- **域名**: www.chanquant.com
- **Web服务器**: Nginx (反向代理)
- **应用服务器**: Uvicorn (FastAPI)
- **端口**: 8000

---

## 🎯 重构目标

### 1. 与重构后的后端集成

将 Web 应用迁移到 `webapp/` 目录，与重构后的 CZSC 分析模块集成：

```
webapp/
├── app/                            # 后端核心模块（已完成）
│   ├── config/
│   ├── core/
│   ├── services/
│   └── utils/
├── web/                            # Web 应用（待重构）
│   ├── __init__.py
│   ├── main.py                     # FastAPI 应用主文件
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py               # API 路由
│   │   ├── models.py               # Pydantic 模型
│   │   └── dependencies.py         # 依赖注入
│   ├── templates/
│   │   ├── index.html
│   │   └── analysis.html
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
├── run.py                          # CLI 工具（已完成）
└── server.py                       # Web 服务器启动脚本（新增）
```

### 2. 模块化重构

**后端模块化**：
- ✅ 配置管理（config）
- ✅ 核心功能（core）
- ✅ 服务层（services）
- ✅ 工具函数（utils）
- ⚠️ Web API（需要重构）

**前端模块化**：
- 拆分路由
- 统一 API 接口
- 优化静态资源

### 3. 功能整合

**当前功能**：
- 股票搜索
- CZSC 分析展示
- 实时追踪
- 智能推荐
- 用户系统

**需要整合**：
- 使用重构后的 `StockAnalyzer`
- 使用重构后的 `DataSourceManager`
- 使用重构后的通知系统

---

## 📝 重构步骤

### 阶段一：Web 应用迁移（1-2天）

#### 1.1 创建 Web 目录结构

```bash
mkdir -p webapp/web/{api,templates,static/{css,js,images}}
touch webapp/web/__init__.py
touch webapp/web/main.py
touch webapp/web/api/{__init__.py,routes.py,models.py,dependencies.py}
```

#### 1.2 迁移 FastAPI 应用

**文件对应**：
- `app.py` → `webapp/web/main.py`
- `api/routes_api.py` → `webapp/web/api/routes.py`
- `api/models.py` → `webapp/web/api/models.py`
- `templates/*` → `webapp/web/templates/*`
- `static/*` → `webapp/web/static/*`

#### 1.3 更新导入路径

```python
# 原来
from analyzer import CZSCAnalysis

# 更新后
from app.services.analyzer import StockAnalyzer
from app.config.settings import get_config
```

### 阶段二：API 路由重构（1天）

#### 2.1 拆分 API 路由

**原来**: `api/routes_api.py` (26KB 单文件)

**重构后**:
```
webapp/web/api/
├── __init__.py
├── routes.py                # 路由注册
├── models.py                # Pydantic 模型
├── dependencies.py          # 依赖注入
└── endpoints/
    ├── __init__.py
    ├── analysis.py          # 分析相关 API
    ├── stock.py             # 股票查询 API
    ├── tracking.py          # 追踪 API
    └── recommend.py         # 推荐 API
```

#### 2.2 API 示例

```python
# webapp/web/api/endpoints/analysis.py
from fastapi import APIRouter, Depends
from app.services.analyzer import StockAnalyzer
from ..models import AnalysisRequest, AnalysisResponse
from ..dependencies import get_analyzer

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/", response_model=AnalysisResponse)
async def analyze_stock(
    request: AnalysisRequest,
    analyzer: StockAnalyzer = Depends(get_analyzer)
):
    """股票分析 API"""
    result = analyzer.analyze_stock(
        request.code,
        request.name,
        request.market,
        request.period
    )
    return AnalysisResponse(**result)
```

### 阶段三：前端优化（1天）

#### 3.1 模板继承

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}缠论量化分析平台{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/main.css">
    {% block extra_css %}{% endblock %}
</head>
<body>
    {% include 'components/header.html' %}
    {% block content %}{% endblock %}
    {% include 'components/footer.html' %}
    {% block extra_js %}{% endblock %}
</body>
</html>
```

#### 3.2 组件化

```
templates/
├── base.html
├── components/
│   ├── header.html
│   ├── footer.html
│   └── stock_card.html
├── pages/
│   ├── index.html
│   ├── analysis.html
│   ├── tracking.html
│   └── recommend.html
└── errors/
    ├── 404.html
    └── 500.html
```

### 阶段四：部署配置（半天）

#### 4.1 服务器启动脚本

```python
# webapp/server.py
import uvicorn
from web.main import app

if __name__ == "__main__":
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

#### 4.2 Nginx 配置

```nginx
# /etc/nginx/sites-available/chanquant.com
server {
    listen 80;
    server_name www.chanquant.com chanquant.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /root/stock_monitor_project/webapp/web/static/;
        expires 30d;
    }
}
```

#### 4.3 systemd 服务

```ini
# /etc/systemd/system/chanquant.service
[Unit]
Description=ChanQuant Web Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/stock_monitor_project/webapp
Environment="PATH=/root/stock_monitor_project/webapp/venv/bin"
ExecStart=/root/stock_monitor_project/webapp/venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 🔧 技术栈

### 后端
- **框架**: FastAPI
- **ASGI 服务器**: Uvicorn
- **数据库**: SQLite (用户数据) + MySQL (K线数据)
- **任务调度**: APScheduler
- **数据分析**: Pandas, NumPy

### 前端
- **模板引擎**: Jinja2
- **图表库**: Plotly.js, Chart.js
- **CSS 框架**: 自定义（可考虑迁移到 Tailwind CSS）
- **JavaScript**: 原生 JS（可考虑 Vue.js）

---

## 📊 API 设计

### RESTful API

#### 股票分析
```
POST /api/analysis
Body: {
    "code": "600000",
    "name": "浦发银行",
    "market": 1,
    "period": "daily"
}
Response: {
    "code": "600000",
    "name": "浦发银行",
    "signals": [...],
    "pivots": [...],
    "divergences": [...]
}
```

#### 股票搜索
```
GET /api/stocks/search?q=浦发
Response: {
    "results": [
        {"code": "600000", "name": "浦发银行", "market": 1}
    ]
}
```

#### 实时追踪
```
GET /api/tracking/status
Response: {
    "tracking_stocks": [...],
    "last_update": "2026-01-09 12:00:00"
}
```

---

## ⚠️ 注意事项

### 1. 兼容性

**原有功能需保留**：
- 用户系统（登录/注册）
- 实时追踪
- 智能推荐
- AI 预测展示

### 2. 数据迁移

**数据库文件**：
- `user_database.db` - 用户数据
- `outputs/*.json` - AI 预测结果

需要确保数据库路径正确。

### 3. 性能优化

- 异步处理耗时操作
- 缓存常用数据
- 静态资源 CDN
- 数据库连接池

---

## 🚀 部署流程

### 1. 本地测试

```bash
cd /root/stock_monitor_project/webapp
python server.py
```

访问：http://localhost:8000

### 2. 生产部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
vim .env

# 3. 启动服务
sudo systemctl start chanquant
sudo systemctl enable chanquant

# 4. 配置 Nginx
sudo ln -s /etc/nginx/sites-available/chanquant.com /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# 5. 配置 SSL（可选）
sudo certbot --nginx -d www.chanquant.com -d chanquant.com
```

---

## 📅 时间估算

| 阶段 | 工作内容 | 预计时间 |
|------|----------|----------|
| 阶段一 | Web 应用迁移 | 1-2 天 |
| 阶段二 | API 路由重构 | 1 天 |
| 阶段三 | 前端优化 | 1 天 |
| 阶段四 | 部署配置 | 0.5 天 |
| **总计** | | **3.5-4.5 天** |

---

## 🎯 下一步行动

### 立即行动

1. **备份原始 Web 应用**
   ```bash
   cd /root/stock_monitor_project
   tar -czf web_backup_$(date +%Y%m%d).tar.gz app.py api/ templates/ static/ run_server.sh
   ```

2. **创建 Web 目录结构**
   ```bash
   cd /root/stock_monitor_project/webapp
   mkdir -p web/{api/endpoints,templates/{components,pages,errors},static/{css,js,images}}
   ```

3. **开始迁移主文件**
   - 将 `app.py` 迁移到 `web/main.py`
   - 更新导入路径
   - 测试基本功能

### 需要你确认

1. **是否需要保留原有的所有功能？**
   - 用户系统？
   - AI 预测展示？
   - 实时追踪？

2. **前端是否需要升级？**
   - 保持原有样式？
   - 还是重新设计？

3. **部署方式？**
   - 继续使用 Nginx + Uvicorn？
   - 还是使用 Docker？

4. **域名和 SSL？**
   - www.chanquant.com 是否仍然可用？
   - SSL 证书是否已配置？

---

**准备好了吗？告诉我你的选择，我们可以立即开始 Web 应用的重构！** 🚀
