#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统回测脚本
基于历史数据回测推荐策略的胜率
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

logger = get_logger("backtest")


def get_trading_days(start_date, end_date):
    """
    获取交易日列表
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        交易日列表
    """
    # 简化版：假设周一到周五都是交易日
    # 实际应该从数据库查询
    trading_days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 周一到周五
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


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
    """找到最近的未闭环买点"""
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


def get_historical_kline(stock_code, target_date, analyzer):
    """
    获取历史K线数据（截止到目标日期）
    
    Args:
        stock_code: 股票代码
        target_date: 目标日期
        analyzer: 分析器
        
    Returns:
        DataFrame: K线数据
    """
    # 从数据库获取所有数据
    data = analyzer.data_source.stock_dao.get_daily_kline(stock_code, limit=500)
    if data is None or len(data) == 0:
        return None
    
    # 筛选目标日期之前的数据
    data.index = pd.to_datetime(data.index)
    target_datetime = pd.Timestamp(target_date)
    filtered_data = data[data.index <= target_datetime]
    
    # 只保留最近300条
    if len(filtered_data) > 300:
        filtered_data = filtered_data.iloc[-300:]
    
    return filtered_data


def backtest_single_day(stock_df, target_date, analyzer):
    """
    回测单个交易日的推荐
    
    Args:
        stock_df: 股票列表DataFrame
        target_date: 目标日期
        analyzer: 股票分析器
        
    Returns:
        推荐列表
    """
    logger.info(f"回测日期: {target_date.strftime('%Y-%m-%d')}")
    
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
            # 获取历史K线（只到目标日期）
            historical_data = get_historical_kline(code, target_date, analyzer)
            if historical_data is None or len(historical_data) < 30:
                continue
            
            # 使用历史数据进行分析（模拟那天的情况）
            # 直接调用分析器的内部方法
            result = analyzer._analyze_with_data(code, name, market, historical_data)
            
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
            
            # 找未闭环买点
            latest_buy = find_latest_unclosed_buy(buy_signals, sell_signals)
            if latest_buy:
                market_name = classify_market(code)
                recommendations.append({
                    "code": code,
                    "name": name,
                    "market": market,
                    "market_name": market_name,
                    "推荐日期": target_date.strftime("%Y-%m-%d"),
                    "推荐价格": latest_buy.get('price'),
                    "推荐类型": latest_buy.get('type'),
                    "信号时间": latest_buy.get('date').strftime("%Y-%m-%d") if latest_buy.get('date') else ""
                })
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            if fail_count < 10:  # 只打印前10个错误
                logger.error(f"❌ {code} - {name} 分析失败: {e}")
    
    logger.info(f"回测完成: 成功={success_count}, 失败={fail_count}, 推荐={len(recommendations)}")
    
    # 按市场分类，每个市场取前20个
    market_recommendations = defaultdict(list)
    for rec in recommendations:
        market_recommendations[rec['market_name']].append(rec)
    
    # 截取每个市场的前20个
    TOP_N = 20
    final_recommendations = []
    for market, recs in market_recommendations.items():
        sorted_recs = sorted(recs, key=lambda x: x['推荐日期'], reverse=True)[:TOP_N]
        final_recommendations.extend(sorted_recs)
    
    return final_recommendations


def calculate_actual_returns(recommendations, analyzer, days):
    """
    计算推荐的实际收益
    
    Args:
        recommendations: 推荐列表
        analyzer: 分析器
        days: 持有天数
        
    Returns:
        带收益的推荐列表
    """
    results = []
    
    for rec in recommendations:
        code = rec['code']
        market = rec['market']
        recommend_date = datetime.strptime(rec['推荐日期'], '%Y-%m-%d')
        recommend_price = rec['推荐价格']
        
        # 计算目标日期（推荐日期 + N天）
        target_date = recommend_date + timedelta(days=days)
        
        try:
            # 获取目标日期的价格
            result = analyzer.analyze_stock(
                code, rec['name'], market,
                period='daily',
                end_date=target_date
            )
            
            if result and 'last_price' in result:
                target_price = result['last_price']
                
                # 计算收益
                if recommend_price > 0:
                    profit_rate = (target_price - recommend_price) / recommend_price * 100
                else:
                    profit_rate = 0
                
                rec[f'{days}天后价格'] = target_price
                rec[f'{days}天收益率'] = round(profit_rate, 2)
                rec['是否盈利'] = profit_rate > 0
            else:
                rec[f'{days}天后价格'] = 0
                rec[f'{days}天收益率'] = 0
                rec['是否盈利'] = False
                
        except Exception as e:
            logger.error(f"计算收益失败 {code}: {e}")
            rec[f'{days}天后价格'] = 0
            rec[f'{days}天收益率'] = 0
            rec['是否盈利'] = False
        
        results.append(rec)
    
    return results


def calculate_win_rate(recommendations_with_returns, days):
    """
    计算胜率统计
    
    Args:
        recommendations_with_returns: 带收益的推荐列表
        days: 天数
        
    Returns:
        胜率统计字典
    """
    total = len(recommendations_with_returns)
    if total == 0:
        return {
            "天数": days,
            "总数": 0,
            "盈利数": 0,
            "亏损数": 0,
            "胜率": 0,
            "平均收益率": 0,
            "最大收益": 0,
            "最大亏损": 0
        }
    
    profit_count = sum(1 for r in recommendations_with_returns if r.get('是否盈利', False))
    loss_count = total - profit_count
    
    profit_rates = [r.get(f'{days}天收益率', 0) for r in recommendations_with_returns]
    avg_profit_rate = sum(profit_rates) / total
    max_profit = max(profit_rates) if profit_rates else 0
    max_loss = min(profit_rates) if profit_rates else 0
    
    win_rate = (profit_count / total * 100) if total > 0 else 0
    
    return {
        "天数": days,
        "总数": total,
        "盈利数": profit_count,
        "亏损数": loss_count,
        "胜率": round(win_rate, 2),
        "平均收益率": round(avg_profit_rate, 2),
        "最大收益": round(max_profit, 2),
        "最大亏损": round(max_loss, 2)
    }


def run_backtest(backtest_days=30):
    """
    运行回测
    
    Args:
        backtest_days: 回测天数（默认30天）
    """
    logger.info("=" * 60)
    logger.info("开始回测推荐系统...")
    logger.info(f"回测天数: {backtest_days}")
    logger.info("=" * 60)
    
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
    
    # 随机采样，加快回测速度（可选）
    # stock_df = stock_df.sample(n=min(500, len(stock_df)), random_state=42)
    # logger.info(f"采样后: {len(stock_df)} 只股票")
    
    # 初始化分析器
    analyzer = StockAnalyzer(use_mysql=True)
    
    # 计算回测日期（30天前）
    today = datetime.now().date()
    backtest_date = today - timedelta(days=backtest_days)
    logger.info(f"回测日期: {backtest_date}")
    
    # 1. 生成那天的推荐
    logger.info("=" * 60)
    logger.info("步骤1: 生成历史推荐...")
    logger.info("=" * 60)
    recommendations = backtest_single_day(stock_df, backtest_date, analyzer)
    logger.info(f"生成推荐数: {len(recommendations)}")
    
    # 2. 计算不同时间窗口的收益
    logger.info("=" * 60)
    logger.info("步骤2: 计算实际收益...")
    logger.info("=" * 60)
    
    all_results = {}
    
    for days in [7, 14, 30]:
        logger.info(f"计算 {days} 天收益...")
        recs_with_returns = calculate_actual_returns(recommendations, analyzer, days)
        win_rate_stats = calculate_win_rate(recs_with_returns, days)
        
        all_results[f'{days}天'] = {
            'recommendations': recs_with_returns,
            'stats': win_rate_stats
        }
        
        logger.info(f"{days}天胜率: {win_rate_stats['胜率']}%")
        logger.info(f"  总数: {win_rate_stats['总数']}")
        logger.info(f"  盈利: {win_rate_stats['盈利数']}")
        logger.info(f"  亏损: {win_rate_stats['亏损数']}")
        logger.info(f"  平均收益: {win_rate_stats['平均收益率']}%")
        logger.info(f"  最大收益: {win_rate_stats['最大收益']}%")
        logger.info(f"  最大亏损: {win_rate_stats['最大亏损']}%")
    
    # 3. 保存回测结果
    logger.info("=" * 60)
    logger.info("步骤3: 保存回测结果...")
    logger.info("=" * 60)
    
    output_path = project_root / "web" / "static" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    backtest_report = {
        "回测日期": backtest_date.strftime("%Y-%m-%d"),
        "回测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "回测天数": backtest_days,
        "推荐总数": len(recommendations),
        "胜率统计": {
            "7天": all_results['7天']['stats'],
            "14天": all_results['14天']['stats'],
            "30天": all_results['30天']['stats']
        },
        "详细推荐": {
            "7天": all_results['7天']['recommendations'][:20],  # 只保存前20个
            "14天": all_results['14天']['recommendations'][:20],
            "30天": all_results['30天']['recommendations'][:20]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(backtest_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"回测结果已保存: {output_path}")
    
    # 4. 生成总结报告
    logger.info("=" * 60)
    logger.info("回测总结:")
    logger.info("=" * 60)
    logger.info(f"回测日期: {backtest_date}")
    logger.info(f"推荐总数: {len(recommendations)}")
    logger.info("")
    
    for days in [7, 14, 30]:
        stats = all_results[f'{days}天']['stats']
        logger.info(f"{days}天表现:")
        logger.info(f"  胜率: {stats['胜率']}%")
        logger.info(f"  平均收益: {stats['平均收益率']}%")
        logger.info(f"  最佳推荐收益: {stats['最大收益']}%")
        logger.info(f"  最差推荐收益: {stats['最大亏损']}%")
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("回测完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 默认回测30天前
    backtest_days = 30
    if len(sys.argv) > 1:
        backtest_days = int(sys.argv[1])
    
    run_backtest(backtest_days)
