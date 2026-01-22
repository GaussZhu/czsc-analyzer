#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看signals数据结构
"""

import sys
sys.path.insert(0, '/root/stock_monitor_project/webapp')

from app.services.analyzer import StockAnalyzer
import json


def check_signals_structure():
    """查看signals的实际数据结构"""
    print("\n" + "="*60)
    print("检查signals数据结构")
    print("="*60)
    
    analyzer = StockAnalyzer(use_mysql=True)
    
    # 分析一只股票
    result = analyzer.analyze_stock(
        stock_code='600000',
        stock_name='浦发银行',
        market=1,
        period='daily'
    )
    
    if result and 'signals' in result:
        signals = result['signals']
        
        print("\n【signals 结构】")
        print(f"Type: {type(signals)}")
        print(f"Keys: {list(signals.keys())}")
        
        print("\n【buy 信号】")
        print(f"Type: {type(signals['buy'])}")
        print(f"Keys: {list(signals['buy'].keys())}")
        
        for key, value in signals['buy'].items():
            print(f"\n  [{key}]")
            print(f"    Type: {type(value)}")
            print(f"    Count: {len(value) if value else 0}")
            if value:
                print(f"    Sample: {value[0] if len(value) > 0 else 'N/A'}")
        
        print("\n【sell 信号】")
        print(f"Type: {type(signals['sell'])}")
        print(f"Keys: {list(signals['sell'].keys())}")
        
        for key, value in signals['sell'].items():
            print(f"\n  [{key}]")
            print(f"    Type: {type(value)}")
            print(f"    Count: {len(value) if value else 0}")
            if value:
                print(f"    Sample: {value[0] if len(value) > 0 else 'N/A'}")
    
    else:
        print("没有找到signals数据")


if __name__ == '__main__':
    check_signals_structure()
