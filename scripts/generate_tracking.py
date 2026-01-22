#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30分钟追踪数据生成脚本
根据推荐列表中的买点，生成追踪数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import time
import pandas as pd
from datetime import datetime

from app.services.analyzer import StockAnalyzer
from app.config.logging_config import get_logger

logger = get_logger("tracking")


def guess_market_from_code(code: str) -> int:
    """根据股票代码推测市场"""
    code = str(code)
    if code.startswith("6"):
        return 1  # 沪市
    elif code.startswith(("0", "3")):
        return 0  # 深市
    elif code.startswith(("83", "87", "88", "89", "92", "43")):
        return 2  # 北交所
    else:
        return 0


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始生成追踪数据...")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 读取推荐数据
    recommend_path = project_root / "web" / "static" / "recommend_result.json"
    if not recommend_path.exists():
        logger.error(f"推荐文件不存在: {recommend_path}")
        return
    
    with open(recommend_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 提取所有买点推荐
    tracking_targets = []
    for block in raw_data.values():
        tracking_targets.extend(block.get("买点推荐", []))
    
    logger.info(f"📊 需要跟踪的买点数量: {len(tracking_targets)}")
    
    # 初始化分析器
    analyzer = StockAnalyzer(use_mysql=True)
    
    tracking_results = []
    success_count = 0
    fail_count = 0
    
    for idx, item in enumerate(tracking_targets):
        code = item["code"]
        name = item["name"]
        buy_time = item["时间"]
        buy_type = item["类型"]
        buy_price = item["价格"]
        
        logger.info(f"[{idx+1}/{len(tracking_targets)}] 🔍 {code} - {name} @ {buy_time}")
        
        try:
            market = guess_market_from_code(code)
            
            # 获取30分钟数据
            data_30min = analyzer.data_source.get_stock_data(
                stock_code=code,
                market=market,
                period='30min',
                count=800,
                prefer_mysql=True
            )
            
            if data_30min is None or data_30min.empty:
                logger.warning(f"⚠️ {code} 无30分钟数据")
                fail_count += 1
                continue
            
            # 计算追踪指标
            buy_date = pd.to_datetime(buy_time)
            
            # 筛选买入时间之后的数据
            tracking_data = data_30min[data_30min.index > buy_date]
            
            if tracking_data.empty:
                # 如果没有买入后的数据，使用最新价格
                current_price = data_30min['收盘'].iloc[-1]
                max_price = current_price
                min_price = current_price
                latest_time = data_30min.index[-1]
            else:
                current_price = tracking_data['收盘'].iloc[-1]
                max_price = tracking_data['最高'].max()
                min_price = tracking_data['最低'].min()
                latest_time = tracking_data.index[-1]
            
            # 计算收益指标
            current_profit = round((current_price - buy_price) / buy_price * 100, 2)
            max_profit = round((max_price - buy_price) / buy_price * 100, 2)
            max_drawdown = round((min_price - buy_price) / buy_price * 100, 2) if min_price < buy_price else 0
            
            # 检查是否有卖点信号
            result = analyzer.analyze_stock(code, name, market, period='daily')
            signals = result.get('signals', []) if result else []
            sell_signals = [s for s in signals if '卖' in s.get('type', '') and s.get('date') > buy_date]
            
            # 构建追踪记录
            row = {
                "name": name,
                "code": code,
                "buy_time": buy_time,
                "buy_type": buy_type,
                "buy_price": buy_price,
                "closed": False,
                "sell_time": "跟踪中",
                "sell_type": "待定",
                "sell_price": "待确认",
                "max_profit": max_profit,
                "current_profit": current_profit,
                "max_drawdown": max_drawdown,
                "latest_time": latest_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 如果有卖点信号，标记为已闭环
            if sell_signals:
                last_sell = sell_signals[-1]
                row["closed"] = True
                row["sell_time"] = last_sell.get("date").strftime("%Y-%m-%d")
                row["sell_type"] = last_sell.get("type")
                row["sell_price"] = last_sell.get("price")
            
            tracking_results.append(row)
            success_count += 1
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"❌ 跟踪失败: {code} - {name}, 原因: {e}")
            fail_count += 1
    
    # 保存结果
    output_path = project_root / "web" / "static" / "tracking_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tracking_results, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("追踪数据生成完成！")
    logger.info(f"成功: {success_count} 条")
    logger.info(f"失败: {fail_count} 条")
    logger.info(f"输出文件: {output_path}")
    
    # 统计
    closed_count = sum(1 for r in tracking_results if r["closed"])
    tracking_count = len(tracking_results) - closed_count
    avg_profit = sum(r["current_profit"] for r in tracking_results) / len(tracking_results) if tracking_results else 0
    
    logger.info(f"跟踪中: {tracking_count} 个")
    logger.info(f"已闭环: {closed_count} 个")
    logger.info(f"平均收益: {avg_profit:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
