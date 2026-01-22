#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统回测脚本 - 简化版
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


def get_price_at_date(stock_code, target_date, analyzer):
    """
    获取指定日期的收盘价
    
    Args:
        stock_code: 股票代码
        target_date: 目标日期
        analyzer: 分析器
        
    Returns:
        float: 收盘价
    """
    try:
        # 获取K线数据
        data = analyzer.data_source.stock_dao.get_daily_kline(stock_code, limit=100)
        if data is None or len(data) == 0:
            return None
        
        # 将date列转换为datetime类型
        data['date'] = pd.to_datetime(data['date'])
        target_datetime = pd.Timestamp(target_date)
        
        # 查找目标日期或之后最近的日期
        future_data = data[data['date'] >= target_datetime]
        if len(future_data) > 0:
            # 返回最接近的日期的收盘价
            return future_data.iloc[0]['close']
        
        return None
        
    except Exception as e:
        logger.error(f"获取价格失败 {stock_code} @ {target_date}: {e}")
        return None


def load_current_recommendations():
    """
    加载当前的推荐结果
    
    Returns:
        list: 推荐列表
    """
    rec_path = project_root / "web" / "static" / "recommend_result.json"
    if not rec_path.exists():
        logger.error(f"推荐文件不存在: {rec_path}")
        return []
    
    with open(rec_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 展开所有市场的推荐
    all_recommendations = []
    for market, recs in data.items():
        for rec in recs.get('买点推荐', []):
            rec['市场'] = market
            all_recommendations.append(rec)
    
    return all_recommendations


def calculate_backtest_returns(recommendations, analyzer, backtest_days):
    """
    计算回测收益
    
    Args:
        recommendations: 推荐列表
        analyzer: 分析器
        backtest_days: 回测天数
        
    Returns:
        dict: 回测结果
    """
    logger.info(f"开始计算 {backtest_days} 天前推荐的实际收益...")
    
    # 计算目标日期（推荐日期 = 今天 - backtest_days天）
    today = datetime.now().date()
    recommend_date = today - timedelta(days=backtest_days)
    
    results_by_period = {}
    
    for hold_days in [7, 14, 30]:
        logger.info(f"计算持有 {hold_days} 天的收益...")
        
        # 目标卖出日期
        sell_date = recommend_date + timedelta(days=hold_days)
        
        results = []
        success_count = 0
        
        for rec in recommendations:
            code = rec['code']
            name = rec['name']
            recommend_price = rec.get('价格', rec.get('当前价格', 0))
            
            if recommend_price == 0:
                continue
            
            # 获取卖出日期的价格
            sell_price = get_price_at_date(code, sell_date, analyzer)
            
            if sell_price is not None and sell_price > 0:
                profit_rate = (sell_price - recommend_price) / recommend_price * 100
                
                results.append({
                    'code': code,
                    'name': name,
                    '市场': rec.get('市场', ''),
                    '推荐价格': recommend_price,
                    f'{hold_days}天后价格': sell_price,
                    f'{hold_days}天收益率': round(profit_rate, 2),
                    '是否盈利': bool(profit_rate > 0),  # 转换为Python bool
                    '推荐类型': rec.get('类型', ''),
                    '推荐时间': rec.get('时间', '')
                })
                success_count += 1
        
        logger.info(f"  成功计算 {success_count}/{len(recommendations)} 只股票")
        
        # 计算统计数据
        if len(results) > 0:
            profit_count = sum(1 for r in results if r['是否盈利'])
            loss_count = len(results) - profit_count
            win_rate = (profit_count / len(results) * 100)
            
            profit_rates = [r[f'{hold_days}天收益率'] for r in results]
            avg_profit = sum(profit_rates) / len(profit_rates)
            max_profit = max(profit_rates)
            max_loss = min(profit_rates)
            
            # 按收益率排序，获取最佳和最差推荐
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
                "最佳推荐": best_picks,
                "最差推荐": worst_picks
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
    
    return {
        '推荐日期': recommend_date.strftime('%Y-%m-%d'),
        '回测时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '回测天数': backtest_days,
        '推荐总数': len(recommendations),
        '结果': results_by_period
    }


def run_simple_backtest(backtest_days=30):
    """
    运行简化版回测（基于当前推荐列表）
    
    Args:
        backtest_days: 回测天数
    """
    logger.info("=" * 60)
    logger.info("简化版回测系统")
    logger.info(f"假设 {backtest_days} 天前使用了相同的推荐策略")
    logger.info("=" * 60)
    
    # 1. 加载当前推荐
    logger.info("加载当前推荐...")
    recommendations = load_current_recommendations()
    logger.info(f"当前推荐数: {len(recommendations)}")
    
    if len(recommendations) == 0:
        logger.error("没有推荐数据！")
        return
    
    # 2. 初始化分析器
    logger.info("初始化分析器...")
    analyzer = StockAnalyzer(use_mysql=True)
    
    # 3. 计算回测收益
    backtest_results = calculate_backtest_returns(recommendations, analyzer, backtest_days)
    
    # 4. 保存结果
    logger.info("=" * 60)
    logger.info("保存回测结果...")
    logger.info("=" * 60)
    
    output_path = project_root / "web" / "static" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"回测结果已保存: {output_path}")
    
    # 5. 打印总结
    logger.info("=" * 60)
    logger.info("回测总结")
    logger.info("=" * 60)
    logger.info(f"假设推荐日期: {backtest_results['推荐日期']}")
    logger.info(f"推荐总数: {backtest_results['推荐总数']}")
    logger.info("")
    
    for period, data in backtest_results['结果'].items():
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
    # 默认回测30天前
    backtest_days = 30
    if len(sys.argv) > 1:
        backtest_days = int(sys.argv[1])
    
    run_simple_backtest(backtest_days)
