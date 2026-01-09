#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL连接测试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db_connector import MySQLConnector
from app.core.stock_dao import StockDAO
from app.config.settings import get_config


def test_connection():
    """测试MySQL连接"""
    print("="*80)
    print("MySQL连接测试")
    print("="*80)
    
    config = get_config()
    
    print(f"\n数据库配置:")
    
    # 检查是否使用Unix socket
    unix_socket = config.mysql.unix_socket
    if unix_socket:
        print(f"  Unix Socket: {unix_socket}")
    else:
        print(f"  Host: {config.mysql.host}:{config.mysql.port}")
    print(f"  Database: {config.mysql.database}")
    print(f"  User: {config.mysql.user}")
    
    # 创建连接
    print(f"\n正在连接...")
    connector = MySQLConnector(
        host=config.mysql.host,
        port=config.mysql.port,
        user=config.mysql.user,
        password=config.mysql.password,
        database=config.mysql.database,
        unix_socket=unix_socket
    )
    
    if not connector.connect():
        print("✗ 连接失败")
        return False
    
    print("✓ 连接成功\n")
    
    # 测试查询
    print("="*80)
    print("测试数据库表")
    print("="*80)
    
    # 显示所有表
    tables_query = "SHOW TABLES"
    tables = connector.execute_query(tables_query)
    
    if tables:
        print(f"\n找到 {len(tables)} 个表:")
        for table in tables:
            table_name = list(table.values())[0]
            print(f"  - {table_name}")
    
    # 测试DAO
    print("\n" + "="*80)
    print("测试StockDAO")
    print("="*80)
    
    dao = StockDAO(connector)
    
    # 获取股票列表
    print("\n1. 获取股票列表（前10个）:")
    stocks = dao.get_all_stock_codes()
    if stocks:
        for i, stock in enumerate(stocks[:10], 1):
            print(f"  {i}. {stock.get('code')} - {stock.get('name')}")
        print(f"  ... 共 {len(stocks)} 只股票")
    
    # 测试获取K线数据
    if stocks:
        test_code = stocks[0]['code']
        print(f"\n2. 获取 {test_code} 的日线数据（最近10条）:")
        
        daily_data = dao.get_daily_kline(test_code, limit=10)
        if daily_data is not None and not daily_data.empty:
            print(f"  ✓ 成功获取 {len(daily_data)} 条记录")
            print(f"\n  数据预览:")
            print(daily_data.head())
            
            # 检查数据范围
            data_range = dao.get_data_range('daily_kline', test_code)
            if data_range:
                print(f"\n  数据范围:")
                print(f"    最早: {data_range['min_date']}")
                print(f"    最新: {data_range['max_date']}")
                print(f"    总数: {data_range['count']}")
        else:
            print(f"  ✗ 未获取到数据")
        
        # 测试30分钟数据
        print(f"\n3. 获取 {test_code} 的30分钟数据（最近5条）:")
        min30_data = dao.get_30min_kline(test_code, limit=5)
        if min30_data is not None and not min30_data.empty:
            print(f"  ✓ 成功获取 {len(min30_data)} 条记录")
        else:
            print(f"  ⚠ 未获取到30分钟数据（可能该股票没有分钟数据）")
    
    # 断开连接
    connector.disconnect()
    print("\n" + "="*80)
    print("✓ 测试完成")
    print("="*80)
    
    return True


def main():
    try:
        success = test_connection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
