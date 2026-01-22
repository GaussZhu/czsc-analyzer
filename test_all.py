#!/usr/bin/env python3
"""
CZSC v2.1 完整测试脚本
测试后端和 Web 应用的所有核心功能
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

def test_backend_imports():
    """测试后端模块导入"""
    print_header("1. 后端模块导入测试")
    
    modules_to_test = [
        ("app.config.settings", "配置管理"),
        ("app.utils.logger", "日志系统"),
        ("app.database.mysql_connector", "MySQL连接器"),
        ("app.dao.stock_dao", "Stock DAO"),
        ("app.services.stock_analyzer", "股票分析器"),
        ("app.services.data_source_manager", "数据源管理器"),
        ("app.models.stock", "股票模型"),
        ("app.czsc.fenxing", "分型识别"),
        ("app.czsc.bi", "笔识别"),
        ("app.czsc.zhongshu", "中枢识别"),
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

def test_config_loading():
    """测试配置加载"""
    print_header("2. 配置系统测试")
    
    try:
        from app.config.settings import settings
        
        print_info(f"MySQL Host: {settings.MYSQL_HOST}")
        print_info(f"MySQL Database: {settings.MYSQL_DATABASE}")
        print_info(f"TDX Host: {settings.TDX_HOST}")
        print_info(f"TDX Port: {settings.TDX_PORT}")
        print_info(f"日志级别: {settings.LOG_LEVEL}")
        
        print_success("配置加载成功")
        return True
    except Exception as e:
        print_error(f"配置加载失败: {e}")
        return False

def test_mysql_connection():
    """测试 MySQL 连接"""
    print_header("3. MySQL 连接测试")
    
    try:
        from app.database.mysql_connector import MySQLConnector
        
        connector = MySQLConnector()
        print_success("MySQL 连接器初始化成功")
        
        # 测试获取连接
        conn = connector.get_connection()
        print_success("获取数据库连接成功")
        
        # 测试查询
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        cursor.close()
        
        print_info(f"数据库中的表数量: {len(tables)}")
        for table in tables:
            print(f"  • {table[0]}")
        
        connector.close()
        print_success("MySQL 连接测试通过")
        return True
        
    except Exception as e:
        print_error(f"MySQL 连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stock_dao():
    """测试 Stock DAO"""
    print_header("4. Stock DAO 测试")
    
    try:
        from app.dao.stock_dao import StockDAO
        
        dao = StockDAO()
        print_success("StockDAO 初始化成功")
        
        # 测试获取股票列表（只取前5条）
        stocks = dao.get_stock_list(limit=5)
        print_info(f"获取到 {len(stocks)} 条股票数据")
        
        if stocks:
            stock = stocks[0]
            print(f"\n示例股票: {stock.code} - {stock.name}")
            
            # 测试获取日线数据
            klines = dao.get_kline_data(stock.code, freq='D', limit=10)
            print_info(f"获取到 {len(klines)} 条日线数据")
            
            if klines:
                latest = klines[0]
                print(f"最新K线: {latest.trade_date} 收盘价={latest.close}")
        
        print_success("Stock DAO 测试通过")
        return True
        
    except Exception as e:
        print_error(f"Stock DAO 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stock_analyzer():
    """测试股票分析器"""
    print_header("5. 股票分析器测试")
    
    try:
        from app.services.stock_analyzer import StockAnalyzer
        
        analyzer = StockAnalyzer()
        print_success("StockAnalyzer 初始化成功")
        
        # 测试分析浦发银行
        test_code = "600000"
        print_info(f"正在分析股票: {test_code}")
        
        result = analyzer.analyze_stock(test_code, freq='D', limit=100)
        
        if result:
            print_info(f"股票名称: {result['stock_info']['name']}")
            print_info(f"K线数量: {result['klines_count']}")
            print_info(f"分型数量: {result['fenxing_count']}")
            print_info(f"笔数量: {result['bi_count']}")
            print_info(f"中枢数量: {result['zhongshu_count']}")
            
            print_success("股票分析测试通过")
            return True
        else:
            print_error("分析结果为空")
            return False
        
    except Exception as e:
        print_error(f"股票分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_app():
    """测试 Web 应用"""
    print_header("6. Web 应用测试")
    
    try:
        from web.main import app
        
        print_success(f"Web 应用初始化成功")
        print_info(f"应用标题: {app.title}")
        print_info(f"应用版本: {app.version}")
        
        # 获取路由列表
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = ','.join(route.methods) if route.methods else 'GET'
                routes.append(f"{methods:15} {route.path}")
        
        print_info(f"注册的路由数量: {len(routes)}")
        print("\n路由列表:")
        for route in sorted(routes):
            print(f"  {route}")
        
        print_success("Web 应用测试通过")
        return True
        
    except Exception as e:
        print_error(f"Web 应用测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """测试 API 端点导入"""
    print_header("7. API 端点测试")
    
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
    
    print(f"\n端点导入统计: {success_count}/{len(endpoints)} 成功")
    return success_count == len(endpoints)

def test_czsc_algorithm():
    """测试 CZSC 算法"""
    print_header("8. CZSC 算法测试")
    
    try:
        from app.czsc.fenxing import FenXingIdentifier
        from app.czsc.bi import BiIdentifier
        from app.czsc.zhongshu import ZhongShuIdentifier
        
        print_success("CZSC 算法模块导入成功")
        
        # 创建测试数据
        from app.dao.stock_dao import StockDAO
        dao = StockDAO()
        
        test_code = "600000"
        klines = dao.get_kline_data(test_code, freq='D', limit=100)
        
        if not klines:
            print_error("无法获取测试数据")
            return False
        
        print_info(f"测试数据: {len(klines)} 条K线")
        
        # 测试分型识别
        fenxing_identifier = FenXingIdentifier()
        fenxing_list = fenxing_identifier.identify(klines)
        print_info(f"识别到 {len(fenxing_list)} 个分型")
        
        # 测试笔识别
        bi_identifier = BiIdentifier()
        bi_list = bi_identifier.identify(fenxing_list)
        print_info(f"识别到 {len(bi_list)} 个笔")
        
        # 测试中枢识别
        zhongshu_identifier = ZhongShuIdentifier()
        zhongshu_list = zhongshu_identifier.identify(bi_list)
        print_info(f"识别到 {len(zhongshu_list)} 个中枢")
        
        print_success("CZSC 算法测试通过")
        return True
        
    except Exception as e:
        print_error(f"CZSC 算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "CZSC v2.1 完整测试套件" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("后端模块导入", test_backend_imports),
        ("配置系统", test_config_loading),
        ("MySQL 连接", test_mysql_connection),
        ("Stock DAO", test_stock_dao),
        ("股票分析器", test_stock_analyzer),
        ("Web 应用", test_web_app),
        ("API 端点", test_api_endpoints),
        ("CZSC 算法", test_czsc_algorithm),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} 执行异常: {e}")
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
        print("\n🎉 所有测试通过！系统运行正常！\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查相关模块\n")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
