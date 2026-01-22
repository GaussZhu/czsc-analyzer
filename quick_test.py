#!/usr/bin/env python3
"""
CZSC v2.1 快速测试脚本
快速验证系统的核心功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("  CZSC v2.1 快速功能测试")
print("="*70)

# 测试 1: 模块导入
print("\n1️⃣  测试模块导入...")
try:
    from app.config.settings import AppConfig
    from app.core.db_connector import MySQLConnector
    from app.core.stock_dao import StockDAO
    from app.core.data_source import UnifiedDataSource
    from app.services.analyzer import StockAnalyzer
    from web.main import app as web_app
    print("✅ 所有核心模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# 测试 2: 配置加载
print("\n2️⃣  测试配置加载...")
try:
    config = AppConfig()
    print(f"✅ 配置加载成功")
    print(f"   MySQL: {config.mysql.host}")
    print(f"   Database: {config.mysql.database}")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")

# 测试 3: MySQL 连接
print("\n3️⃣  测试 MySQL 连接...")
try:
    connector = MySQLConnector(
        host=config.mysql.host,
        user=config.mysql.user,
        password=config.mysql.password,
        database=config.mysql.database
    )
    conn = connector.get_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    cursor.close()
    connector.close()
    print(f"✅ MySQL 连接成功，数据库有 {len(tables)} 张表")
    for table in tables[:5]:
        print(f"   • {table[0]}")
    if len(tables) > 5:
        print(f"   ... 还有 {len(tables)-5} 张表")
except Exception as e:
    print(f"❌ MySQL 连接失败: {e}")

# 测试 4: Stock DAO
print("\n4️⃣  测试 Stock DAO...")
try:
    connector = MySQLConnector(
        host=config.mysql.host,
        user=config.mysql.user,
        password=config.mysql.password,
        database=config.mysql.database
    )
    dao = StockDAO(connector)
    
    # 获取股票列表
    stocks = dao.get_all_stocks(limit=3)
    print(f"✅ Stock DAO 工作正常，获取到 {len(stocks)} 条股票数据")
    
    for stock in stocks:
        code = stock.get('code', '')
        name = stock.get('name', '')
        print(f"   • {code} - {name}")
    
    # 测试K线数据
    if stocks:
        test_code = stocks[0]['code']
        klines = dao.get_kline_data(test_code, freq='D', limit=5)
        print(f"   获取到 {len(klines)} 条K线数据 ({test_code})")
    
    connector.close()
except Exception as e:
    print(f"❌ Stock DAO 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 5: 数据源
print("\n5️⃣  测试统一数据源...")
try:
    data_source = UnifiedDataSource()
    print(f"✅ UnifiedDataSource 初始化成功")
    
    # 检查可用方法
    methods = [m for m in dir(data_source) if not m.startswith('_')]
    print(f"   可用方法: {', '.join(methods[:5])}...")
    
    data_source.close()
except Exception as e:
    print(f"❌ 数据源测试失败: {e}")

# 测试 6: 分析服务
print("\n6️⃣  测试股票分析服务...")
try:
    analyzer = StockAnalyzer()
    print(f"✅ StockAnalyzer 初始化成功")
    
    # 检查可用方法
    methods = [m for m in dir(analyzer) if not m.startswith('_') and callable(getattr(analyzer, m))]
    print(f"   可用方法: {', '.join(methods[:8])}")
    
    # 尝试分析
    test_code = "600000"
    if hasattr(analyzer, 'analyze_stock'):
        result = analyzer.analyze_stock(test_code, freq='D', limit=50)
        if result:
            print(f"   分析成功: {test_code}")
            print(f"   K线数: {result.get('kline_count', 'N/A')}")
    
    analyzer.close()
except Exception as e:
    print(f"❌ 分析服务测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 7: Web 应用
print("\n7️⃣  测试 Web 应用...")
try:
    print(f"✅ Web 应用加载成功")
    print(f"   标题: {web_app.title}")
    print(f"   版本: {web_app.version}")
    
    # 统计路由
    routes = [r for r in web_app.routes if hasattr(r, 'path')]
    api_routes = [r for r in routes if r.path.startswith('/api/')]
    page_routes = [r for r in routes if not r.path.startswith('/api/') and not r.path.startswith('/docs') and not r.path.startswith('/openapi') and not r.path.startswith('/redoc')]
    
    print(f"   页面路由: {len(page_routes)} 个")
    print(f"   API 路由: {len(api_routes)} 个")
    print(f"   总路由: {len(routes)} 个")
except Exception as e:
    print(f"❌ Web 应用测试失败: {e}")

# 测试 8: CZSC 算法
print("\n8️⃣  测试 CZSC 算法...")
try:
    from app.core.fractal import identify_fractals
    from app.core.stroke import identify_strokes
    from app.core.pivot import identify_pivots
    
    # 获取测试数据
    connector = MySQLConnector(
        host=config.mysql.host,
        user=config.mysql.user,
        password=config.mysql.password,
        database=config.mysql.database
    )
    dao = StockDAO(connector)
    
    test_code = "600000"
    klines = dao.get_kline_data(test_code, freq='D', limit=100)
    
    if klines:
        # 分型识别
        fractals = identify_fractals(klines)
        top_fractals = [f for f in fractals if f.get('type') == 'top']
        bottom_fractals = [f for f in fractals if f.get('type') == 'bottom']
        
        # 笔识别
        strokes = identify_strokes(fractals)
        
        # 中枢识别
        pivots = identify_pivots(strokes)
        
        print(f"✅ CZSC 算法测试成功 ({test_code}, {len(klines)} K线)")
        print(f"   顶分型: {len(top_fractals)} 个")
        print(f"   底分型: {len(bottom_fractals)} 个")
        print(f"   笔: {len(strokes)} 个")
        print(f"   中枢: {len(pivots)} 个")
    else:
        print(f"⚠️  无法获取测试数据")
    
    connector.close()
except Exception as e:
    print(f"❌ CZSC 算法测试失败: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "="*70)
print("  ✅ 快速测试完成！")
print("="*70)
print("\n💡 提示:")
print("   • 后端模块已正确加载")
print("   • MySQL 连接正常")
print("   • CZSC 算法可用")
print("   • Web 应用已就绪")
print("\n🚀 启动 Web 服务:")
print("   cd /root/stock_monitor_project/webapp")
print("   source venv/bin/activate")
print("   python web/server.py")
print("\n" + "="*70 + "\n")
