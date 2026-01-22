#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整回测系统
使用历史K线数据运行推荐算法，验证策略有效性
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict

from app.services.analyzer import StockAnalyzer
from app.config.logging_config import get_logger

logger = get_logger("backtest_full")


def classify_market(code: str) -> str:
    """市场归类"""
    code = str(code)
    if code.startswith("688") or code.startswith("689"):
        return "科创板"
    elif code.startswith("60"):
        return "上证"
    elif code.startswith("00"):
        return "深证"
    elif code.startswith("30"):
        return "创业板"
    elif code.startswith(("83", "87", "92", "43")):
        return "北交所"
    else:
        return "其他"


def find_latest_unclosed_buy(buy_signals, sell_signals):
    """找到最近的未闭环买点（已废弃，保留兼容性）"""
    if not buy_signals:
        return None
    buy_signals_sorted = sorted(buy_signals, key=lambda x: x.get("date"), reverse=True)
    for buy_sig in buy_signals_sorted:
        buy_date = buy_sig.get("date")
        has_later_sell = any(
            sell_sig.get("date") > buy_date 
            for sell_sig in sell_signals
        )
        if not has_later_sell:
            return buy_sig
    return None


def find_today_buy_signals(buy_signals, target_date):
    """
    找到指定日期当天出现的买点信号
    
    Args:
        buy_signals: 买点信号列表
        target_date: 目标日期（datetime对象）
        
    Returns:
        当天的买点信号列表（可能有多个）
    """
    if not buy_signals:
        return []
    
    # 确保是date对象
    if hasattr(target_date, 'date'):
        target_date = target_date.date()
    
    # 筛选出当天的买点
    today_buys = []
    for buy_sig in buy_signals:
        buy_date = buy_sig.get("date")
        if buy_date and hasattr(buy_date, 'date'):
            buy_date = buy_date.date()
        
        if buy_date == target_date:
            today_buys.append(buy_sig)
    
    # 按类型排序：first_type > second_type > third_type
    type_priority = {'first_type': 1, 'second_type': 2, 'third_type': 3}
    today_buys_sorted = sorted(
        today_buys, 
        key=lambda x: type_priority.get(x.get('type', 'third_type'), 999)
    )
    
    return today_buys_sorted


def generate_recommendations_at_date(stock_df, target_date, analyzer, top_n=20):
    """
    生成指定日期的推荐（使用历史数据）
    
    Args:
        stock_df: 股票列表
        target_date: 目标日期
        analyzer: 分析器
        top_n: 每个市场取多少个推荐
        
    Returns:
        推荐列表
    """
    logger.info(f"=" * 60)
    logger.info(f"生成 {target_date} 的历史推荐...")
    logger.info(f"=" * 60)
    
    recommendations = []
    success_count = 0
    fail_count = 0
    
    for idx, row in stock_df.iterrows():
        code = str(row['code']).zfill(6)
        name = row['name']
        market = int(row.get('market', 0))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"进度: {idx+1}/{len(stock_df)}")
        
        try:
            # 关键：使用 analyze_stock_with_date_filter，只分析目标日期之前的数据
            result = analyzer.analyze_stock_with_date_filter(
                code, name, market,
                end_date=target_date,
                period='daily'
            )
            
            if not result or 'signals' not in result:
                continue
            
            signals = result['signals']
            last_price = result.get('last_price', 0)
            
            # 提取买点信号
            buy_signals = []
            if isinstance(signals, dict):
                for signal_type, points in signals.get('buy', {}).items():
                    for p in points:
                        buy_signals.append({
                            'type': p.get('type', signal_type),
                            'price': p.get('price'),
                            'date': p.get('time')
                        })
            
            sell_signals = []
            if isinstance(signals, dict):
                for signal_type, points in signals.get('sell', {}).items():
                    for p in points:
                        sell_signals.append({
                            'type': p.get('type', signal_type),
                            'price': p.get('price'),
                            'date': p.get('time')
                        })
            
            # 🔴 新逻辑：只推荐目标日期当天出现的买点
            today_buys = find_today_buy_signals(buy_signals, target_date)
            for buy_signal in today_buys:
                market_name = classify_market(code)
                recommendations.append({
                    "code": code,
                    "name": name,
                    "market": market,
                    "market_name": market_name,
                    "推荐日期": target_date.strftime("%Y-%m-%d"),
                    "推荐价格": buy_signal.get('price'),
                    "推荐类型": buy_signal.get('type'),
                    "信号时间": buy_signal.get('date').strftime("%Y-%m-%d") if buy_signal.get('date') else ""
                })
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            if fail_count <= 10:  # 只打印前10个错误
                logger.error(f"❌ {code} - {name} 分析失败: {e}")
    
    logger.info(f"分析完成: 成功={success_count}, 失败={fail_count}")
    
    # 按市场分类，每个市场取前N个
    market_recommendations = defaultdict(list)
    for rec in recommendations:
        market_recommendations[rec['market_name']].append(rec)
    
    final_recommendations = []
    for market, recs in market_recommendations.items():
        sorted_recs = sorted(recs, key=lambda x: x['推荐日期'], reverse=True)[:top_n]
        final_recommendations.extend(sorted_recs)
        logger.info(f"{market}: {len(sorted_recs)} 个推荐")
    
    logger.info(f"总推荐数: {len(final_recommendations)}")
    return final_recommendations


def get_price_at_date(stock_code, target_date, analyzer):
    """获取指定日期的收盘价"""
    try:
        data = analyzer.data_source.stock_dao.get_daily_kline(stock_code, limit=100)
        if data is None or len(data) == 0:
            return None
        
        data['date'] = pd.to_datetime(data['date'])
        target_datetime = pd.Timestamp(target_date)
        
        # 查找目标日期或之后最近的日期
        future_data = data[data['date'] >= target_datetime]
        if len(future_data) > 0:
            return future_data.iloc[0]['close']
        
        return None
        
    except Exception as e:
        logger.error(f"获取价格失败 {stock_code} @ {target_date}: {e}")
        return None


def calculate_returns(recommendations, analyzer, hold_days_list):
    """
    计算推荐的实际收益
    
    Args:
        recommendations: 推荐列表
        analyzer: 分析器
        hold_days_list: 持有天数列表，如[7, 14, 30]
        
    Returns:
        各持有期的收益统计
    """
    logger.info(f"=" * 60)
    logger.info(f"计算实际收益...")
    logger.info(f"=" * 60)
    
    recommend_date = datetime.strptime(recommendations[0]['推荐日期'], '%Y-%m-%d').date()
    
    results_by_period = {}
    
    for hold_days in hold_days_list:
        logger.info(f"计算持有 {hold_days} 天的收益...")
        
        sell_date = recommend_date + timedelta(days=hold_days)
        
        results = []
        for rec in recommendations:
            code = rec['code']
            recommend_price = rec['推荐价格']
            
            if recommend_price == 0:
                continue
            
            # 获取卖出日期的价格
            sell_price = get_price_at_date(code, sell_date, analyzer)
            
            if sell_price is not None and sell_price > 0:
                profit_rate = (sell_price - recommend_price) / recommend_price * 100
                
                results.append({
                    'code': code,
                    'name': rec['name'],
                    '市场': rec.get('market_name', ''),
                    '推荐价格': recommend_price,
                    f'{hold_days}天后价格': sell_price,
                    f'{hold_days}天收益率': round(profit_rate, 2),
                    '是否盈利': bool(profit_rate > 0),
                    '推荐类型': rec.get('推荐类型', ''),
                    '信号时间': rec.get('信号时间', '')
                })
        
        logger.info(f"  成功计算 {len(results)}/{len(recommendations)} 只股票")
        
        # 计算统计数据
        if len(results) > 0:
            profit_count = sum(1 for r in results if r['是否盈利'])
            loss_count = len(results) - profit_count
            win_rate = (profit_count / len(results) * 100)
            
            profit_rates = [r[f'{hold_days}天收益率'] for r in results]
            avg_profit = sum(profit_rates) / len(profit_rates)
            max_profit = max(profit_rates)
            max_loss = min(profit_rates)
            
            # 按收益率排序
            sorted_results = sorted(results, key=lambda x: x[f'{hold_days}天收益率'], reverse=True)
            best_picks = sorted_results[:5]
            worst_picks = sorted_results[-5:]
            
            stats = {
                "天数": hold_days,
                "总数": len(results),
                "盈利数": profit_count,
                "亏损数": loss_count,
                "胜率": round(win_rate, 2),
                "平均收益率": round(avg_profit, 2),
                "最大收益": round(max_profit, 2),
                "最大亏损": round(max_loss, 2),
                "最佳推荐": best_picks[:5],
                "最差推荐": worst_picks[:5]
            }
            
            logger.info(f"  胜率: {stats['胜率']}%")
            logger.info(f"  平均收益: {stats['平均收益率']}%")
            logger.info(f"  最大收益: {stats['最大收益']}%")
            logger.info(f"  最大亏损: {stats['最大亏损']}%")
        else:
            stats = {
                "天数": hold_days,
                "总数": 0,
                "盈利数": 0,
                "亏损数": 0,
                "胜率": 0,
                "平均收益率": 0,
                "最大收益": 0,
                "最大亏损": 0,
                "最佳推荐": [],
                "最差推荐": []
            }
        
        results_by_period[f'{hold_days}天'] = {
            'stats': stats,
            'all_results': results
        }
    
    return results_by_period


def run_full_backtest(backtest_date_str, hold_days_list=[7, 14, 30], sample_size=None):
    """
    运行完整回测
    
    Args:
        backtest_date_str: 回测日期字符串，如 "2025-12-20"
        hold_days_list: 持有天数列表
        sample_size: 采样数量（用于加快测试）
    """
    logger.info("=" * 60)
    logger.info("完整回测系统")
    logger.info(f"回测日期: {backtest_date_str}")
    logger.info("=" * 60)
    
    # 转换日期
    backtest_date = datetime.strptime(backtest_date_str, '%Y-%m-%d').date()
    
    # 加载股票列表
    stock_list_path = project_root / "all_stock_list.csv"
    if not stock_list_path.exists():
        stock_list_path = project_root.parent / "all_stock_list.csv"
    
    if not stock_list_path.exists():
        logger.error(f"股票列表文件不存在: {stock_list_path}")
        return
    
    logger.info(f"加载股票列表: {stock_list_path}")
    stock_df = pd.read_csv(stock_list_path, dtype={'code': str, 'name': str})
    logger.info(f"共 {len(stock_df)} 只股票")
    
    # 采样（可选，加快测试）
    if sample_size and sample_size < len(stock_df):
        stock_df = stock_df.sample(n=sample_size, random_state=42)
        logger.info(f"采样后: {len(stock_df)} 只股票")
    
    # 初始化分析器
    analyzer = StockAnalyzer(use_mysql=True)
    
    # 1. 生成历史推荐
    recommendations = generate_recommendations_at_date(stock_df, backtest_date, analyzer, top_n=20)
    
    if len(recommendations) == 0:
        logger.error("未生成任何推荐！")
        return
    
    # 2. 计算实际收益
    results_by_period = calculate_returns(recommendations, analyzer, hold_days_list)
    
    # 3. 保存结果
    logger.info("=" * 60)
    logger.info("保存回测结果...")
    logger.info("=" * 60)
    
    output_path = project_root / "web" / "static" / "backtest_full_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    backtest_report = {
        '回测方法': 'FULL',
        '回测日期': backtest_date_str,
        '回测时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '推荐总数': len(recommendations),
        '采样数量': sample_size if sample_size else len(stock_df),
        '结果': results_by_period,
        '历史推荐': recommendations[:20]  # 只保存前20个
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(backtest_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"回测结果已保存: {output_path}")
    
    # 4. 打印总结
    logger.info("=" * 60)
    logger.info("回测总结")
    logger.info("=" * 60)
    logger.info(f"回测日期: {backtest_date_str}")
    logger.info(f"推荐总数: {len(recommendations)}")
    logger.info("")
    
    for period, data in results_by_period.items():
        stats = data['stats']
        logger.info(f"{period}表现:")
        logger.info(f"  成功回测: {stats['总数']} 只")
        logger.info(f"  胜率: {stats['胜率']}%")
        logger.info(f"  平均收益: {stats['平均收益率']}%")
        logger.info(f"  最大收益: {stats['最大收益']}%")
        logger.info(f"  最大亏损: {stats['最大亏损']}%")
        
        if stats['最佳推荐']:
            best = stats['最佳推荐'][0]
            days_key = f"{stats['天数']}天收益率"
            logger.info(f"  最佳推荐: {best['name']}({best['code']}) +{best[days_key]}%")
        
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("回测完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 默认回测2025-12-20
    backtest_date = "2025-12-20"
    sample_size = None  # 设置为数字可以采样测试，如500
    
    if len(sys.argv) > 1:
        backtest_date = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    
    run_full_backtest(backtest_date, hold_days_list=[7, 14, 30], sample_size=sample_size)
