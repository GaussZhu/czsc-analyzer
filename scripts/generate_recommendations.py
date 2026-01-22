#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐信号生成脚本
每天扫描全市场，生成买点和卖点推荐列表
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import time
import json
from collections import defaultdict
from datetime import datetime

from app.services.analyzer import StockAnalyzer
from app.config.logging_config import get_logger

logger = get_logger("recommendations")


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
    """
    找到最近的未闭环买点（已废弃，保留兼容性）
    """
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


def find_today_buy_signals(buy_signals, target_date=None):
    """
    找到指定日期当天出现的买点信号
    
    Args:
        buy_signals: 买点信号列表
        target_date: 目标日期（datetime对象），如果为None则使用今天
        
    Returns:
        当天的买点信号列表（可能有多个）
    """
    if not buy_signals:
        return []
    
    # 如果没有指定日期，使用今天
    if target_date is None:
        target_date = datetime.now().date()
    else:
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


def find_latest_unpaired_sell(sell_signals, buy_signals):
    """
    找到最近的孤立卖点（之前没有买点）
    
    Args:
        sell_signals: 卖点信号列表
        buy_signals: 买点信号列表
        
    Returns:
        最近的孤立卖点，如果没有则返回None
    """
    if not sell_signals:
        return None
    
    # 按时间排序（最新的在前）
    sell_signals_sorted = sorted(sell_signals, key=lambda x: x.get("date"), reverse=True)
    
    for sell_sig in sell_signals_sorted:
        sell_date = sell_sig.get("date")
        
        # 检查这个卖点之前是否有买点
        has_earlier_buy = any(
            buy_sig.get("date") < sell_date 
            for buy_sig in buy_signals
        )
        
        if not has_earlier_buy:
            return sell_sig
    
    return None


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始生成推荐信号...")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # 初始化分析器
    analyzer = StockAnalyzer(use_mysql=True)
    
    # 结果字典
    recommendations = defaultdict(lambda: {"买点推荐": [], "卖点推荐": []})
    
    # 扫描所有股票
    success_count = 0
    fail_count = 0
    
    for idx, row in stock_df.iterrows():
        code = str(row['code']).zfill(6)
        name = row['name']
        market = int(row.get('market', 0))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"进度: {idx+1}/{len(stock_df)}")
        
        try:
            # 分析股票
            result = analyzer.analyze_stock(code, name, market, period='daily')
            
            if not result or 'signals' not in result:
                continue
            
            signals = result['signals']
            last_price = result.get('last_price', 0)
            
            # 信号是dict格式，需要转换为list
            buy_signals = []
            if isinstance(signals, dict):
                # 从buy字典中提取所有买点
                for signal_type, points in signals.get('buy', {}).items():
                    for p in points:
                        buy_signals.append({
                            'type': p.get('type', signal_type),
                            'price': p.get('price'),
                            'date': p.get('time')
                        })
            
            sell_signals = []
            if isinstance(signals, dict):
                # 从sell字典中提取所有卖点
                for signal_type, points in signals.get('sell', {}).items():
                    for p in points:
                        sell_signals.append({
                            'type': p.get('type', signal_type),
                            'price': p.get('price'),
                            'date': p.get('time')
                        })
            
            # 市场分类
            market_name = classify_market(code)
            
            # 🔴 新逻辑：只推荐今天出现的买点
            today_buys = find_today_buy_signals(buy_signals, target_date=None)
            for buy_signal in today_buys:
                recommendations[market_name]["买点推荐"].append({
                    "code": code,
                    "name": name,
                    "类型": buy_signal.get('type'),
                    "价格": buy_signal.get('price'),
                    "当前价格": last_price,
                    "时间": buy_signal.get('date').strftime("%Y-%m-%d"),
                    "推荐理由": f"{buy_signal.get('type')}信号确认，今日出现买点"
                })
            
            # 找孤立卖点
            latest_sell = find_latest_unpaired_sell(sell_signals, buy_signals)
            if latest_sell:
                recommendations[market_name]["卖点推荐"].append({
                    "code": code,
                    "name": name,
                    "类型": latest_sell.get('type'),
                    "价格": latest_sell.get('price'),
                    "当前价格": last_price,
                    "时间": latest_sell.get('date').strftime("%Y-%m-%d"),
                    "推荐理由": f"{latest_sell.get('type')}信号确认，注意风险控制"
                })
            
            success_count += 1
            time.sleep(0.05)  # 避免过快
            
        except Exception as e:
            logger.error(f"❌ {code} - {name} 分析失败: {e}")
            fail_count += 1
    
    # 截取每个市场的前20个买点和卖点（扩大推荐池，避免错杀）
    TOP_N = 20  # 可配置的推荐数量
    for market in recommendations:
        recommendations[market]["买点推荐"] = sorted(
            recommendations[market]["买点推荐"], 
            key=lambda x: x["时间"], 
            reverse=True
        )[:TOP_N]
        recommendations[market]["卖点推荐"] = sorted(
            recommendations[market]["卖点推荐"], 
            key=lambda x: x["时间"], 
            reverse=True
        )[:TOP_N]
    
    # 保存结果
    output_path = project_root / "web" / "static" / "recommend_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("推荐信号生成完成！")
    logger.info(f"成功: {success_count} 只")
    logger.info(f"失败: {fail_count} 只")
    logger.info(f"输出文件: {output_path}")
    
    # 统计推荐数量
    total_buy = sum(len(v["买点推荐"]) for v in recommendations.values())
    total_sell = sum(len(v["卖点推荐"]) for v in recommendations.values())
    logger.info(f"买点推荐: {total_buy} 个")
    logger.info(f"卖点推荐: {total_sell} 个")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
