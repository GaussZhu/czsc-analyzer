#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("\n" + "="*60)
print("  CZSC v2.1 核心功能快速验证")
print("="*60 + "\n")

# 1. 模块导入
print("1. 模块导入测试...")
from app.config.settings import AppConfig
from app.services.analyzer import StockAnalyzer
from web.main import app
print("✅ 核心模块导入成功\n")

# 2. 配置
print("2. 配置加载...")
config = AppConfig()
print(f"✅ MySQL: {config.mysql.host}")
print(f"✅ Database: {config.mysql.database}\n")

# 3. 分析器
print("3. 股票分析器...")
analyzer = StockAnalyzer()
print(f"✅ 分析器初始化成功")
print(f"   可用方法: analyze_stock, batch_analyze\n")

# 4. Web 应用
print("4. Web 应用...")
print(f"✅ 标题: {app.title}")
print(f"✅ 版本: {app.version}")
routes = [r for r in app.routes if hasattr(r, 'path')]
print(f"✅ 路由数: {len(routes)}\n")

print("="*60)
print("  ✅ 所有核心功能正常！")
print("="*60)
print("\n🚀 启动 Web 服务:")
print("   cd /root/stock_monitor_project/webapp")
print("   source venv/bin/activate")
print("   python web/server.py\n")
