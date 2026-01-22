#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多级别共振分析系统
"""

import sys
sys.path.insert(0, '/root/stock_monitor_project/webapp')

from app.services.multi_level_resonance import (
    MultiLevelResonanceAnalyzer,
    format_resonance_report,
    TimeLevel
)


def test_single_stock():
    """测试单只股票"""
    print("\n" + "="*60)
    print("【测试1】单只股票多级别共振分析")
    print("="*60)
    
    analyzer = MultiLevelResonanceAnalyzer(use_mysql=True)
    
    # 测试浦发银行（只使用支持的级别：daily和30min）
    signal = analyzer.analyze_stock(
        code='600000',
        name='浦发银行',
        market=1,
        levels=[TimeLevel.DAILY, TimeLevel.MIN30]  # 移除MIN5
    )
    
    if signal:
        print("\n✅ 发现共振信号！")
        print(format_resonance_report(signal))
    else:
        print("\n❌ 未发现共振信号")
    
    analyzer.close()


def test_market_scan():
    """测试全市场扫描"""
    print("\n" + "="*60)
    print("【测试2】全市场共振信号扫描")
    print("="*60)
    
    analyzer = MultiLevelResonanceAnalyzer(use_mysql=True)
    
    # 测试股票列表（从当前推荐列表中选择）
    test_stocks = [
        ('600000', '浦发银行', 1),
        ('000001', '平安银行', 0),
        ('300552', '万集科技', 0),
        ('300198', 'ST纳川', 0),
        ('300498', '温氏股份', 0),
        ('300177', '中海达', 0),
        ('601166', '兴业银行', 1),
        ('600036', '招商银行', 1),
        ('000002', '万科A', 0),
        ('600519', '贵州茅台', 1),
    ]
    
    print(f"\n开始扫描 {len(test_stocks)} 只股票...")
    
    resonances = analyzer.scan_market(
        stock_list=test_stocks,
        min_resonance_level=2  # 至少2级共振
    )
    
    print(f"\n扫描完成！发现 {len(resonances)} 个共振信号\n")
    
    if resonances:
        print("="*60)
        print("【共振信号汇总】（按置信度排序）")
        print("="*60)
        
        for i, signal in enumerate(resonances, 1):
            print(f"\n{i}. {signal.code} {signal.name}")
            print(f"   • 共振级别: {signal.resonance_level}级")
            print(f"   • 置信度: {signal.confidence:.1%}")
            print(f"   • 预期收益: +{signal.expected_profit:.1f}%")
            print(f"   • 盈亏比: {signal.risk_reward:.2f}:1")
            print(f"   • 建议入场价: ¥{signal.entry_price:.2f}")
            print(f"   • 止损价: ¥{signal.stop_loss:.2f}")
            
            buy_types = [f"{p.level.value}:{p.buy_type.value}" for p in signal.buy_points]
            print(f"   • 买点组合: {', '.join(buy_types)}")
        
        # 详细报告（只显示前3个）
        print("\n" + "="*60)
        print("【详细报告】（Top 3）")
        print("="*60)
        
        for signal in resonances[:3]:
            print(format_resonance_report(signal))
    
    else:
        print("\n未发现任何共振信号")
    
    analyzer.close()


def test_resonance_levels():
    """测试不同共振级别的筛选"""
    print("\n" + "="*60)
    print("【测试3】不同共振级别筛选")
    print("="*60)
    
    analyzer = MultiLevelResonanceAnalyzer(use_mysql=True)
    
    test_stocks = [
        ('600000', '浦发银行', 1),
        ('000001', '平安银行', 0),
        ('300552', '万集科技', 0),
    ]
    
    # 测试只要三级共振
    print("\n【筛选条件】只显示三级共振信号:")
    resonances_3 = analyzer.scan_market(
        stock_list=test_stocks,
        min_resonance_level=3
    )
    
    print(f"结果: {len(resonances_3)} 个三级共振信号")
    for signal in resonances_3:
        print(f"  • {signal.code} {signal.name} - 置信度: {signal.confidence:.1%}")
    
    # 测试两级及以上共振
    print("\n【筛选条件】显示两级及以上共振信号:")
    resonances_2 = analyzer.scan_market(
        stock_list=test_stocks,
        min_resonance_level=2
    )
    
    print(f"结果: {len(resonances_2)} 个共振信号")
    for signal in resonances_2:
        print(f"  • {signal.code} {signal.name} - {signal.resonance_level}级 - 置信度: {signal.confidence:.1%}")
    
    analyzer.close()


def main():
    """主测试函数"""
    print("\n" + "🎯"*30)
    print("缠论区间套多级别共振分析系统 - 测试")
    print("🎯"*30)
    
    try:
        # 测试1: 单只股票
        test_single_stock()
        
        # 测试2: 全市场扫描
        test_market_scan()
        
        # 测试3: 不同级别筛选
        test_resonance_levels()
        
        print("\n" + "="*60)
        print("✅ 所有测试完成！")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
