#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日推荐跟踪脚本
跟踪每天推荐的股票表现，计算胜率
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

logger = get_logger("daily_tracking")


def load_recommendations():
    """加载今天的推荐"""
    rec_path = project_root / "web" / "static" / "recommend_result.json"
    if not rec_path.exists():
        logger.error(f"推荐文件不存在: {rec_path}")
        return None
    
    with open(rec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tracking_history():
    """加载历史跟踪数据"""
    tracking_path = project_root / "web" / "static" / "tracking_history.json"
    if not tracking_path.exists():
        return []
    
    with open(tracking_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tracking_history(data):
    """保存跟踪历史"""
    tracking_path = project_root / "web" / "static" / "tracking_history.json"
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_win_rate(tracking_data, days):
    """
    计算指定天数的胜率
    
    Args:
        tracking_data: 跟踪数据列表
        days: 天数（7, 14, 30等）
        
    Returns:
        胜率字典
    """
    today = datetime.now().date()
    target_date = today - timedelta(days=days)
    
    # 筛选出指定天数前的推荐
    relevant = [
        t for t in tracking_data
        if datetime.strptime(t['推荐日期'], '%Y-%m-%d').date() <= target_date
    ]
    
    if not relevant:
        return {"胜率": 0, "总数": 0, "盈利": 0, "亏损": 0, "平均涨跌幅": 0}
    
    # 统计盈利和亏损
    profit_count = sum(1 for t in relevant if t.get('涨跌幅', 0) > 0)
    loss_count = sum(1 for t in relevant if t.get('涨跌幅', 0) <= 0)
    total = len(relevant)
    
    win_rate = (profit_count / total * 100) if total > 0 else 0
    
    return {
        "胜率": round(win_rate, 2),
        "总数": total,
        "盈利": profit_count,
        "亏损": loss_count,
        "平均涨跌幅": round(sum(t.get('涨跌幅', 0) for t in relevant) / total, 2) if total > 0 else 0
    }


def update_tracking():
    """更新跟踪数据"""
    logger.info("=" * 60)
    logger.info("开始更新每日跟踪...")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 加载推荐
    recommendations = load_recommendations()
    if not recommendations:
        return
    
    # 加载历史跟踪
    tracking_history = load_tracking_history()
    
    # 今天的日期
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化分析器
    analyzer = StockAnalyzer(use_mysql=True)
    
    # 更新现有跟踪记录的当前价格
    logger.info("更新现有跟踪记录...")
    for track in tracking_history:
        code = track['code']
        market = track.get('market', 0)
        
        try:
            # 获取最新价格
            result = analyzer.analyze_stock(code, track['name'], market, period='daily')
            if result and 'last_price' in result:
                old_price = track.get('当前价格', track['推荐价格'])
                track['当前价格'] = result['last_price']
                
                # 计算涨跌幅
                buy_price = track['推荐价格']
                current_price = track['当前价格']
                track['涨跌幅'] = round((current_price - buy_price) / buy_price * 100, 2)
                
                # 计算持有天数
                recommend_date = datetime.strptime(track['推荐日期'], '%Y-%m-%d')
                track['持有天数'] = (datetime.now() - recommend_date).days
                
                # 更新时间
                track['更新时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
        except Exception as e:
            logger.error(f"更新 {code} 失败: {e}")
    
    # 添加今天的新推荐
    logger.info("添加今天的新推荐...")
    new_count = 0
    for market, data in recommendations.items():
        for buy_rec in data.get("买点推荐", []):
            code = buy_rec['code']
            
            # 检查是否已经跟踪
            if any(t['code'] == code and t['推荐日期'] == today for t in tracking_history):
                continue
            
            # 添加新跟踪
            tracking_history.append({
                "code": code,
                "name": buy_rec['name'],
                "market": 0 if code.startswith("60") else 1,
                "推荐日期": today,
                "推荐价格": buy_rec.get('价格', buy_rec.get('当前价格', 0)),
                "当前价格": buy_rec.get('当前价格', 0),
                "涨跌幅": 0,
                "持有天数": 0,
                "推荐类型": buy_rec.get('类型', ''),
                "推荐理由": buy_rec.get('推荐理由', ''),
                "市场": market,
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            new_count += 1
    
    logger.info(f"新增 {new_count} 条跟踪记录")
    
    # 保存跟踪历史
    save_tracking_history(tracking_history)
    
    # 计算胜率统计
    logger.info("=" * 60)
    logger.info("胜率统计：")
    
    stats_7d = calculate_win_rate(tracking_history, 7)
    stats_14d = calculate_win_rate(tracking_history, 14)
    stats_30d = calculate_win_rate(tracking_history, 30)
    
    logger.info(f"7天胜率: {stats_7d['胜率']}% (总数: {stats_7d['总数']}, 盈利: {stats_7d['盈利']}, 亏损: {stats_7d['亏损']}, 平均涨跌: {stats_7d['平均涨跌幅']}%)")
    logger.info(f"14天胜率: {stats_14d['胜率']}% (总数: {stats_14d['总数']}, 盈利: {stats_14d['盈利']}, 亏损: {stats_14d['亏损']}, 平均涨跌: {stats_14d['平均涨跌幅']}%)")
    logger.info(f"30天胜率: {stats_30d['胜率']}% (总数: {stats_30d['总数']}, 盈利: {stats_30d['盈利']}, 亏损: {stats_30d['亏损']}, 平均涨跌: {stats_30d['平均涨跌幅']}%)")
    
    # 保存胜率统计
    stats_path = project_root / "web" / "static" / "win_rate_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "7天": stats_7d,
            "14天": stats_14d,
            "30天": stats_30d,
            "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("每日跟踪更新完成！")
    logger.info(f"跟踪总数: {len(tracking_history)}")
    logger.info(f"输出文件: {tracking_history}")
    logger.info("=" * 60)


if __name__ == "__main__":
    update_tracking()
