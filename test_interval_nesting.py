#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试区间套分析系统"""

import sys
sys.path.insert(0, '/root/stock_monitor_project/webapp')

from app.services.interval_nesting import IntervalNestingAnalyzer, format_signal_report

def main():
    print("=" * 70)
    print("测试区间套分析系统")
    print("策略: 周线确认趋势 → 日线找买点 → 30分钟/5分钟精确入场")
    print("=" * 70)
    print()
    
    # 初始化分析器
    print("初始化分析器...")
    analyzer = IntervalNestingAnalyzer(use_mysql=True)
    
    # 测试股票
    test_stocks = [
        ('600000', '浦发银行', 1),
        ('000001', '平安银行', 0),
    ]
    
    for code, name, market in test_stocks:
        print(f"\n正在分析: {code} {name}")
        print("-" * 70)
        
        try:
            signal = analyzer.analyze(
                code=code,
                name=name,
                market=market,
                check_intraday=True
            )
            
            if signal:
                print(format_signal_report(signal))
            else:
                print(f"❌ {name} 未发现有效信号（可能周线下跌或日线无买点）")
        
        except Exception as e:
            print(f"⚠️ 分析 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    analyzer.close()
    print("\n测试完成！")

if __name__ == '__main__':
    main()
