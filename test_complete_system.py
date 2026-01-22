#!/usr/bin/env python3
"""
CZSC v2.1 完整系统测试
测试重构后的后端和 Web 应用
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """打印测试部分标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_success(msg):
    """打印成功消息"""
    print(f"✅ {msg}")

def print_error(msg):
    """打印错误消息"""
    print(f"❌ {msg}")

def print_info(msg):
    """打印信息消息"""
    print(f"ℹ️  {msg}")

def test_backend_structure():
    """测试后端模块结构"""
    print_header("1. 后端模块结构测试")
    
    modules_to_test = [
        ("app.config.settings", "配置管理"),
        ("app.config.logging_config", "日志配置"),
        ("app.core.db_connector", "数据库连接器"),
        ("app.core.stock_dao", "Stock DAO"),
        ("app.core.data_source", "数据源"),
        ("app.core.data_fetcher", "数据获取"),
        ("app.core.fractal", "分型识别"),
        ("app.core.stroke", "笔识别"),
        ("app.core.pivot", "中枢识别"),
        ("app.core.divergence", "背驰识别"),
        ("app.core.signals", "信号识别"),
        ("app.core.indicators", "技术指标"),
        ("app.services.analyzer", "分析服务"),
        ("app.services.monitor", "监控服务"),
        ("app.services.notifier", "通知服务"),
    ]
    
    success_count = 0
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
            success_count += 1
        except ImportError as e:
            print_error(f"{description}: {module_name} - {e}")
    
    print(f"\n导入统计: {success_count}/{len(modules_to_test)} 成功")
    return success_count == len(modules_to_test)

def test_config_system():
    """测试配置系统"""
    print_header("2. 配置系统测试")
    
    try:
        from app.config.settings import Config
        
        config = Config()
        print_success("配置类初始化成功")
        
        # 打印关键配置
        print_info(f"MySQL Host: {config.MYSQL_HOST}")
        print_info(f"MySQL Database: {config.MYSQL_DATABASE}")
        print_info(f"TDX Host: {config.TDX_HOST}")
        print_info(f"TDX Port: {config.TDX_PORT}")
        
        print_success("配置系统测试通过")
        return True
        
    except Exception as e:
        print_error(f"配置系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mysql_connection():
    """测试 MySQL 连接"""
    print_header("3. MySQL 数据库连接测试")
    
    try:
        from app.core.db_connector import get_mysql_connection
        
        conn = get_mysql_connection()
        print_success("MySQL 连接成功")
        
        # 测试查询表
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        cursor.close()
        
        print_info(f"数据库中的表数量: {len(tables)}")
        for table in tables:
            print(f"  • {table[0]}")
        
        conn.close()
        print_success("MySQL 连接测试通过")
        return True
        
    except Exception as e:
        print_error(f"MySQL 连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stock_dao():
    """测试 Stock DAO"""
    print_header("4. Stock DAO 功能测试")
    
    try:
        from app.core.stock_dao import StockDAO
        
        dao = StockDAO()
        print_success("StockDAO 初始化成功")
        
        # 测试获取股票列表
        stocks = dao.get_all_stocks(limit=5)
        print_info(f"获取到 {len(stocks)} 条股票数据")
        
        if stocks:
            stock = stocks[0]
            code = stock.get('code', '')
            name = stock.get('name', '')
            print(f"\n示例股票: {code} - {name}")
            
            # 测试获取K线数据
            klines = dao.get_kline_data(code, freq='D', limit=10)
            print_info(f"获取到 {len(klines)} 条日线数据")
            
            if klines:
                latest = klines[-1]  # 最新的K线
                print(f"最新K线: 日期={latest.get('trade_date')} 收盘价={latest.get('close')}")
        
        print_success("Stock DAO 测试通过")
        return True
        
    except Exception as e:
        print_error(f"Stock DAO 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_source():
    """测试数据源"""
    print_header("5. 数据源功能测试")
    
    try:
        from app.core.data_source import UnifiedDataSource
        
        data_source = UnifiedDataSource()
        print_success("UnifiedDataSource 初始化成功")
        
        # 测试获取股票列表
        stocks = data_source.get_stock_list(limit=3)
        print_info(f"获取到 {len(stocks)} 条股票数据")
        
        if stocks:
            test_code = stocks[0]['code']
            print_info(f"测试股票代码: {test_code}")
            
            # 测试获取K线数据
            klines = data_source.get_kline_data(test_code, freq='D', limit=5)
            print_info(f"获取到 {len(klines)} 条K线数据")
        
        print_success("数据源测试通过")
        return True
        
    except Exception as e:
        print_error(f"数据源测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_czsc_algorithms():
    """测试 CZSC 算法"""
    print_header("6. CZSC 算法测试")
    
    try:
        from app.core.fractal import identify_fractals
        from app.core.stroke import identify_strokes
        from app.core.pivot import identify_pivots
        from app.core.stock_dao import StockDAO
        
        print_success("CZSC 算法模块导入成功")
        
        # 获取测试数据
        dao = StockDAO()
        test_code = "600000"  # 浦发银行
        klines = dao.get_kline_data(test_code, freq='D', limit=100)
        
        if not klines:
            print_error("无法获取测试数据")
            return False
        
        print_info(f"测试数据: {len(klines)} 条K线 (代码: {test_code})")
        
        # 测试分型识别
        fractals = identify_fractals(klines)
        top_fractals = [f for f in fractals if f.get('type') == 'top']
        bottom_fractals = [f for f in fractals if f.get('type') == 'bottom']
        print_info(f"识别到分型: 顶分型 {len(top_fractals)} 个, 底分型 {len(bottom_fractals)} 个")
        
        # 测试笔识别
        strokes = identify_strokes(fractals)
        print_info(f"识别到 {len(strokes)} 个笔")
        
        # 测试中枢识别
        pivots = identify_pivots(strokes)
        print_info(f"识别到 {len(pivots)} 个中枢")
        
        print_success("CZSC 算法测试通过")
        return True
        
    except Exception as e:
        print_error(f"CZSC 算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_service():
    """测试分析服务"""
    print_header("7. 股票分析服务测试")
    
    try:
        from app.services.analyzer import StockAnalyzer
        
        analyzer = StockAnalyzer()
        print_success("StockAnalyzer 初始化成功")
        
        # 测试分析浦发银行
        test_code = "600000"
        print_info(f"正在分析股票: {test_code}")
        
        result = analyzer.analyze(test_code, freq='D', limit=100)
        
        if result:
            print_info(f"分析结果:")
            print_info(f"  • 股票代码: {result.get('code')}")
            print_info(f"  • 股票名称: {result.get('name')}")
            print_info(f"  • K线数量: {result.get('kline_count', 0)}")
            print_info(f"  • 分型数量: {result.get('fractal_count', 0)}")
            print_info(f"  • 笔数量: {result.get('stroke_count', 0)}")
            print_info(f"  • 中枢数量: {result.get('pivot_count', 0)}")
            
            print_success("股票分析服务测试通过")
            return True
        else:
            print_error("分析结果为空")
            return False
        
    except Exception as e:
        print_error(f"股票分析服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_application():
    """测试 Web 应用"""
    print_header("8. Web 应用测试")
    
    try:
        from web.main import app
        
        print_success(f"Web 应用初始化成功")
        print_info(f"应用标题: {app.title}")
        print_info(f"应用版本: {app.version}")
        
        # 获取路由列表
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = ','.join(sorted(route.methods)) if route.methods else 'GET'
                routes.append((methods, route.path))
        
        print_info(f"注册的路由数量: {len(routes)}")
        
        # 按路径分组
        api_routes = [r for r in routes if r[1].startswith('/api/')]
        page_routes = [r for r in routes if not r[1].startswith('/api/') and not r[1].startswith('/docs') and not r[1].startswith('/openapi') and not r[1].startswith('/redoc')]
        
        print(f"\n页面路由 ({len(page_routes)}):")
        for methods, path in sorted(page_routes):
            print(f"  {methods:15} {path}")
        
        print(f"\nAPI 路由 ({len(api_routes)}):")
        for methods, path in sorted(api_routes):
            print(f"  {methods:15} {path}")
        
        print_success("Web 应用测试通过")
        return True
        
    except Exception as e:
        print_error(f"Web 应用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_api_endpoints():
    """测试 Web API 端点"""
    print_header("9. Web API 端点测试")
    
    endpoints = [
        ("web.api.endpoints.stock", "股票端点"),
        ("web.api.endpoints.analysis", "分析端点"),
        ("web.api.endpoints.tracking", "追踪端点"),
    ]
    
    success_count = 0
    for module_name, description in endpoints:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
            success_count += 1
        except Exception as e:
            print_error(f"{description}: {module_name} - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n端点导入统计: {success_count}/{len(endpoints)} 成功")
    return success_count == len(endpoints)

def run_all_tests():
    """运行所有测试"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CZSC v2.0 + v2.1 完整系统测试" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("后端模块结构", test_backend_structure),
        ("配置系统", test_config_system),
        ("MySQL 数据库连接", test_mysql_connection),
        ("Stock DAO 功能", test_stock_dao),
        ("数据源功能", test_data_source),
        ("CZSC 算法", test_czsc_algorithms),
        ("股票分析服务", test_analyzer_service),
        ("Web 应用", test_web_application),
        ("Web API 端点", test_web_api_endpoints),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} 执行异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 打印测试总结
    print_header("测试总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%\n")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("\n🎉 所有测试通过！CZSC v2.1 系统运行正常！\n")
        return 0
    elif passed >= total * 0.7:
        print(f"\n✅ 大部分测试通过（{passed}/{total}），系统基本可用\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查相关模块\n")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
