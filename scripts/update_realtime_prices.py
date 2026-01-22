#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时行情更新脚本
使用腾讯财经API获取股票实时价格，更新tracking_history.json
"""

import sys
import os
from pathlib import Path
import json
import requests
import time
from datetime import datetime, time as dt_time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config.logging_config import get_logger

logger = get_logger("realtime_prices")


def is_trading_time():
    """判断是否在交易时间内"""
    now = datetime.now()
    current_time = now.time()
    
    # 周末不交易
    if now.weekday() >= 5:
        return False
    
    # 交易时间：9:30-11:30, 13:00-15:00
    morning_start = dt_time(9, 30)
    morning_end = dt_time(11, 30)
    afternoon_start = dt_time(13, 0)
    afternoon_end = dt_time(15, 0)
    
    in_morning = morning_start <= current_time <= morning_end
    in_afternoon = afternoon_start <= current_time <= afternoon_end
    
    return in_morning or in_afternoon


def get_market_prefix(code):
    """根据股票代码获取市场前缀"""
    code = str(code)
    
    # 上证：600/601/603/688
    if code.startswith(('60', '68')):
        return 'sh'
    # 深证：000/001/002/003
    elif code.startswith('00'):
        return 'sz'
    # 创业板：300/301
    elif code.startswith('30'):
        return 'sz'
    # 科创板：688
    elif code.startswith('688'):
        return 'sh'
    # 北交所：43/83/87/92
    elif code.startswith(('43', '83', '87', '92')):
        return 'bj'
    else:
        return 'sz'  # 默认深圳


def fetch_realtime_prices_tencent(stock_codes):
    """
    使用腾讯财经API获取实时行情
    
    Args:
        stock_codes: 股票代码列表
        
    Returns:
        字典 {code: price}
    """
    if not stock_codes:
        return {}
    
    # 构建查询参数（批量查询，最多50只）
    # 格式：sz000001,sh600000,sz300001
    symbols = [f"{get_market_prefix(code)}{code}" for code in stock_codes]
    
    # 分批查询（每批50只）
    batch_size = 50
    all_prices = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        query_str = ','.join(batch)
        
        url = f"http://qt.gtimg.cn/q={query_str}"
        
        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'gbk'
            
            if response.status_code != 200:
                logger.error(f"获取行情失败: {response.status_code}")
                continue
            
            # 解析返回数据
            # 格式：v_sz000001="51~平安银行~000001~11.13~11.07~11.07~..."
            for line in response.text.strip().split('\n'):
                if not line or '=' not in line:
                    continue
                
                try:
                    # 提取股票代码和数据
                    parts = line.split('=')
                    if len(parts) != 2:
                        continue
                    
                    var_name = parts[0].strip()  # v_sz000001
                    data_str = parts[1].strip().strip('"')  # 数据字符串
                    
                    # 从变量名提取代码
                    code = var_name.split('_')[-1][-6:]  # 提取后6位
                    
                    # 分割数据
                    fields = data_str.split('~')
                    if len(fields) < 4:
                        continue
                    
                    # 字段说明：
                    # 0: 未知
                    # 1: 名称
                    # 2: 代码
                    # 3: 当前价格
                    # 4: 昨收
                    # 5: 今开
                    # ...
                    
                    current_price = float(fields[3])
                    all_prices[code] = current_price
                    
                except Exception as e:
                    logger.error(f"解析行情数据失败: {line[:100]}, 错误: {e}")
                    continue
            
            # 避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
            continue
    
    return all_prices


def update_tracking_prices():
    """更新追踪列表中的实时价格"""
    logger.info("=" * 60)
    logger.info("开始更新实时行情...")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 判断是否在交易时间
    if not is_trading_time():
        logger.info("⏰ 非交易时间，跳过更新")
        return
    
    logger.info("✅ 交易时间内，开始更新")
    logger.info("=" * 60)
    
    # 读取追踪数据
    tracking_file = project_root / "web" / "static" / "tracking_history.json"
    
    if not tracking_file.exists():
        logger.error(f"追踪文件不存在: {tracking_file}")
        return
    
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            tracking_data = json.load(f)
        
        logger.info(f"读取追踪数据: {len(tracking_data)} 条")
        
        if len(tracking_data) == 0:
            logger.info("追踪列表为空，无需更新")
            return
        
        # 提取股票代码
        stock_codes = [item['code'] for item in tracking_data]
        logger.info(f"需要更新的股票: {len(stock_codes)} 只")
        
        # 获取实时价格
        prices = fetch_realtime_prices_tencent(stock_codes)
        logger.info(f"成功获取价格: {len(prices)} 只")
        
        # 更新价格和涨跌幅
        update_count = 0
        for item in tracking_data:
            code = item['code']
            if code in prices:
                old_price = item.get('当前价格', 0)
                new_price = prices[code]
                
                # 更新当前价格
                item['当前价格'] = new_price
                
                # 重新计算涨跌幅
                recommend_price = item.get('推荐价格', 0)
                if recommend_price > 0:
                    profit_rate = (new_price - recommend_price) / recommend_price * 100
                    item['涨跌幅'] = round(profit_rate, 2)
                
                # 更新时间
                item['更新时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                update_count += 1
                
                # 打印价格变化（如果有变化）
                if abs(new_price - old_price) > 0.01:
                    change = new_price - old_price
                    logger.info(f"{code} {item['name']}: {old_price:.2f} -> {new_price:.2f} ({change:+.2f})")
        
        # 保存更新后的数据
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"✅ 更新完成！成功更新 {update_count} 只股票")
        logger.info(f"保存到: {tracking_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"更新失败: {e}", exc_info=True)


def main():
    """主函数"""
    update_tracking_prices()


if __name__ == "__main__":
    main()
