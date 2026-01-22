# Web 应用模块

## 状态

🚧 **待重构** - 将在后续版本中完成

## 原始 Web 应用

原始的 FastAPI Web 应用位于：
- `/root/stock_monitor_project/app.py`
- `/root/stock_monitor_project/templates/`
- `/root/stock_monitor_project/static/`
- `/root/stock_monitor_project/api/`

已部署在: **www.chanquant.com**

## 重构计划

Web 应用将进行以下重构：

1. **迁移到 `webapp/web/` 目录**
2. **整合重构后的后端模块**
   - 使用 `app.services.analyzer.StockAnalyzer`
   - 使用 `app.core.data_source.DataSourceManager`
   - 使用 `app.config.settings` 配置管理

3. **简化用户系统**
   - 移除注册/登录功能
   - 改为**免注册使用**
   - 保留核心分析功能

4. **API 路由模块化**
   - 拆分大文件 `api/routes_api.py` (26KB)
   - 按功能分类：analysis, stock, tracking, recommend

5. **前端优化**
   - 模板继承和组件化
   - 移除登录/注册界面
   - 优化用户体验

## 技术栈

- **框架**: FastAPI
- **模板引擎**: Jinja2
- **图表**: Plotly.js, Chart.js
- **部署**: Nginx + Uvicorn

## 详细计划

请查看: [WEB_REFACTORING_PLAN.md](../WEB_REFACTORING_PLAN.md)

## 时间线

- **v2.0**: 后端模块化重构 ✅ (已完成)
- **v2.1**: Web 应用重构 🚧 (计划中)
- **v3.0**: 全栈整合 📅 (未来)

---

**备注**: 当前版本 (v2.0) 专注于后端核心模块的重构，Web 应用将在 v2.1 版本中完成。
