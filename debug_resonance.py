#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试多级别共振分析 - 查看买点信息
"""

import sys
sys.path.insert(0, '/root/stock_monitor_project/webapp')

from app.services.multi_level_resonance import (
    MultiLevelResonanceAnalyzer,
    TimeLevel
)


def debug_buy_points():
    """调试查看买点信息"""
    print("\n" + "="*60)
    print("调试：查看各级别买点信息")
    print("="*60)
    
    analyzer = MultiLevelResonanceAnalyzer(use_mysql=True)
    
    code = '600000'
    name = '浦发银行'
    market = 1
    
    # 分别查看各级别的买点
    for level in [TimeLevel.DAILY, TimeLevel.MIN30]:
        print(f"\n【{level.value} 级别】")
        print("-" * 60)
        
        points = analyzer._find_buy_points(code, name, market, level)
        
        if points:
            print(f"发现 {len(points)} 个买点：")
            for i, point in enumerate(points, 1):
                print(f"\n  买点 {i}:")
                print(f"    类型: {point.buy_type.value}")
                print(f"    价格: ¥{point.price:.2f}")
                print(f"    时间: {point.time}")
                print(f"    置信度: {point.confidence:.2%}")
                print(f"    理由: {point.reason}")
                print(f"    止损: ¥{point.stop_loss:.2f}")
                print(f"    目标: ¥{point.target_price:.2f}")
        else:
            print("  未发现买点")
    
    # 测试共振检查
    print("\n" + "="*60)
    print("测试共振分析")
    print("="*60)
    
    signal = analyzer.analyze_stock(code, name, market)
    
    if signal:
        print(f"\n✅ 发现 {signal.resonance_level} 级共振！")
        print(f"   置信度: {signal.confidence:.2%}")
        print(f"   预期收益: +{signal.expected_profit:.1f}%")
        print(f"   盈亏比: {signal.risk_reward:.2f}:1")
        
        print(f"\n   共振买点:")
        for point in signal.buy_points:
            print(f"     • [{point.level.value}] {point.buy_type.value} @ ¥{point.price:.2f} ({point.time})")
    else:
        print("\n❌ 未形成共振")
        
        # 看看为什么
        all_points = []
        for level in [TimeLevel.DAILY, TimeLevel.MIN30]:
            points = analyzer._find_buy_points(code, name, market, level)
            all_points.extend(points)
        
        if len(all_points) >= 2:
            print(f"\n   共有 {len(all_points)} 个买点，但未形成共振")
            print("   可能原因：")
            print("     1. 时间窗口不重叠")
            print("     2. 买点级别分布不符合要求")
            
            # 检查时间重叠
            print(f"\n   时间分布:")
            for point in all_points:
                print(f"     • [{point.level.value}] {point.time}")
        else:
            print(f"\n   只有 {len(all_points)} 个买点，不足以形成共振（需要至少2个）")
    
    analyzer.close()


if __name__ == '__main__':
    debug_buy_points()
