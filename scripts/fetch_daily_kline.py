#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线数据定时更新脚本
每天交易日收盘后自动更新日线、周线、30分钟K线数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import json
import pandas as pd
from datetime import datetime
from pytdx.hq import TdxHq_API
from pytdx.util.best_ip import select_best_ip

from app.core.data_source import UnifiedDataSource, DataSourceType
from app.config.logging_config import get_logger

logger = get_logger("fetch_kline")

BESTIP_CACHE_PATH = project_root / "config" / "bestip_cache.json"


def load_last_bestip():
    """加载上次使用的最优IP"""
    if BESTIP_CACHE_PATH.exists():
        try:
            with open(BESTIP_CACHE_PATH, "r") as f:
                bestip = json.load(f)
            if "ip" in bestip and "port" in bestip:
                return bestip
        except Exception as e:
            logger.warning(f"读取bestip缓存失败: {e}")
    return None


def save_bestip(ip_info):
    """保存最优IP到缓存"""
    try:
        BESTIP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BESTIP_CACHE_PATH, "w") as f:
            json.dump(ip_info, f)
        logger.info(f"保存最优IP: {ip_info['ip']}:{ip_info['port']}")
    except Exception as e:
        logger.warning(f"保存bestip缓存失败: {e}")


def connect_best_tdx():
    """连接到最优的TDX服务器"""
    logger.info("尝试连接TDX服务器...")
    
    api = TdxHq_API()
    last_bestip = load_last_bestip()
    
    # 先尝试缓存的IP
    if last_bestip:
        try:
            if api.connect(last_bestip["ip"], last_bestip["port"], time_out=3):
                logger.info(f"✅ 使用缓存IP连接成功: {last_bestip['ip']}:{last_bestip['port']}")
                return api
        except Exception as e:
            logger.warning(f"⚠️ 缓存IP连接失败: {e}")
            api.disconnect()
    
    # 重新检测最优IP
    logger.info("重新检测最优TDX服务器...")
    best = select_best_ip()
    if not best:
        logger.error("❌ 无可用TDX服务器！")
        return None
    
    try:
        if api.connect(best['ip'], best['port'], time_out=3):
            logger.info(f"✅ 重新检测后连接成功: {best['ip']}:{best['port']}")
            save_bestip(best)
            return api
    except Exception as e:
        logger.error(f"❌ 连接最优服务器失败: {e}")
        return None


def guess_market(code):
    """根据股票代码推测市场"""
    code = str(code)
    if code.startswith("6"):
        return 1  # 沪市
    elif code.startswith(("4", "8")) or code.startswith("83") or code.startswith("87") or code.startswith("92"):
        return 2  # 北交所
    else:
        return 0  # 深市


def fetch_and_store_stock(api, data_source, stock_code, stock_name, market, period, count):
    """
    拉取并存储股票K线数据
    
    Args:
        api: TDX API对象
        data_source: 统一数据源对象
        stock_code: 股票代码
        stock_name: 股票名称
        market: 市场代码
        period: 周期类型 (daily/weekly/30min)
        count: 拉取数量
    """
    try:
        period_map = {
            "daily": 9,
            "weekly": 5,
            "30min": 7
        }
        data_type = period_map.get(period)
        
        if data_type is None:
            logger.error(f"未知周期类型: {period}")
            return
        
        # 拉取数据
        data = []
        index = 0
        
        while len(data) < count:
            chunk = api.get_security_bars(data_type, market, stock_code, index, min(800, count - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            index += len(chunk)
            if len(chunk) < 800:
                break
        
        if not data:
            logger.warning(f"⚠️ 拉取 {stock_name}({stock_code}) {period} 数据失败")
            return
        
        # 转换为DataFrame
        columns = ['open', 'close', 'high', 'low', 'vol', 'amount', 'year', 'month', 'day', 'hour', 'minute']
        df = pd.DataFrame(data, columns=columns[:len(data[0])])
        
        # 生成日期
        if period == "30min":
            df['datetime'] = pd.to_datetime(
                df[['year', 'month', 'day', 'hour', 'minute']]
                .astype(str)
                .apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}-{int(x['day']):02d} {int(x['hour']):02d}:{int(x['minute']):02d}:00", axis=1)
            )
        else:
            df['datetime'] = pd.to_datetime(
                df[['year', 'month', 'day']]
                .astype(str)
                .apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}-{int(x['day']):02d}", axis=1)
            )
        
        # 检查已存在的数据
        existing_df = data_source.get_stock_data(
            stock_code=stock_code,
            market=market,
            period=period,
            count=count,
            prefer_mysql=True
        )
        
        if existing_df is not None and not existing_df.empty:
            # 只插入新数据
            existing_dates = set(existing_df.index.astype(str))
            df = df[~df['datetime'].astype(str).isin(existing_dates)]
        
        if df.empty:
            logger.info(f"✅ {stock_name}({stock_code}) {period}: 数据已是最新")
            return
        
        # 重命名列以匹配数据库格式
        df.rename(columns={
            'datetime': '日期',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'vol': '成交量',
            'amount': '成交额'
        }, inplace=True)
        
        # 设置索引
        df.set_index('日期', inplace=True)
        
        # 添加股票信息
        df['code'] = stock_code
        df['name'] = stock_name
        
        # 只保留必要列
        df = df[['code', 'name', '开盘', '最高', '最低', '收盘', '成交量', '成交额']]
        
        # 存储到MySQL
        data_source.save_to_mysql(stock_code, period, df)
        logger.info(f"📈 {stock_name}({stock_code}) {period}: 成功插入 {len(df)} 条新数据")
        
    except Exception as e:
        logger.error(f"❌ {stock_name}({stock_code}) 拉取{period}失败: {e}", exc_info=True)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始更新K线数据...")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 加载股票列表
    stock_list_path = project_root / "all_stock_list.csv"
    if not stock_list_path.exists():
        # 尝试从根目录加载
        stock_list_path = project_root.parent / "all_stock_list.csv"
    
    if not stock_list_path.exists():
        logger.error(f"股票列表文件不存在: {stock_list_path}")
        return
    
    logger.info(f"加载股票列表: {stock_list_path}")
    stock_df = pd.read_csv(stock_list_path, dtype={'code': str, 'name': str})
    logger.info(f"共 {len(stock_df)} 只股票")
    
    # 连接TDX
    api = connect_best_tdx()
    if not api:
        logger.error("无法连接TDX，退出脚本")
        return
    
    # 初始化数据源
    data_source = UnifiedDataSource(DataSourceType.MYSQL)
    
    # 定义更新周期
    periods = [
        {"period": "daily", "count": 600},
        {"period": "weekly", "count": 240},
        {"period": "30min", "count": 800},
    ]
    
    # 更新每只股票的数据
    success_count = 0
    fail_count = 0
    
    for idx, row in stock_df.iterrows():
        code = str(row['code']).zfill(6)
        name = row['name']
        market = row.get('market', guess_market(code))
        
        logger.info(f"\n[{idx+1}/{len(stock_df)}] {name}({code})")
        
        try:
            for p in periods:
                fetch_and_store_stock(
                    api, 
                    data_source,
                    code, 
                    name, 
                    market, 
                    period=p["period"], 
                    count=p["count"]
                )
            success_count += 1
            time.sleep(0.5)  # 防止请求过快
        except Exception as e:
            logger.error(f"处理 {name}({code}) 失败: {e}")
            fail_count += 1
    
    # 清理
    api.disconnect()
    
    # 统计
    logger.info("=" * 60)
    logger.info("K线数据更新完成！")
    logger.info(f"成功: {success_count} 只")
    logger.info(f"失败: {fail_count} 只")
    logger.info(f"总计: {len(stock_df)} 只")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
