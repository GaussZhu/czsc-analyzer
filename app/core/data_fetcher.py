# -*- coding: utf-8 -*-
"""
Data Fetcher Module
负责从通达信服务器获取股票数据
"""

import time
import datetime
import pandas as pd
from pytdx.hq import TdxHq_API
from pytdx.util.best_ip import select_best_ip

from app.config.settings import get_config
from app.config.logging_config import get_data_logger

logger = get_data_logger()


class TDXDataFetcher:
    """通达信数据获取器"""
    
    def __init__(self):
        """初始化数据获取器"""
        self.config = get_config()
        self.api = TdxHq_API()
        self.connected = False
        self.best_ip = None
        self.current_server = None
        
        logger.info("TDX数据获取器初始化完成")
    
    def update_best_server(self):
        """动态更新最优服务器"""
        logger.info("正在检测最优服务器...")
        try:
            best = select_best_ip()
            if best:
                self.best_ip = (best['ip'], best['port'])
                logger.info(f"最优服务器: {best['ip']}:{best['port']}")
                
                # 如果最优服务器不在配置列表中，则插入首位
                if self.best_ip not in self.config.tdx.servers:
                    self.config.tdx.servers.insert(0, self.best_ip)
            else:
                logger.warning("未能检测到最优服务器，使用默认列表")
        except Exception as e:
            logger.error(f"检测最优服务器失败: {str(e)}")
    
    def connect(self) -> bool:
        """
        连接到TDX服务器（智能重试机制）
        
        Returns:
            bool: 连接是否成功
        """
        if self.connected:
            logger.debug("已经连接到TDX服务器")
            return True
        
        # 每次连接前更新最优服务器
        self.update_best_server()
        
        for attempt in range(self.config.tdx.retry_count):
            logger.info(f"第 {attempt + 1}/{self.config.tdx.retry_count} 次连接尝试")
            
            # 优先尝试最优服务器
            for idx, (ip, port) in enumerate(self.config.tdx.servers, 1):
                try:
                    logger.debug(f"尝试服务器 {idx}/{len(self.config.tdx.servers)}: {ip}:{port}")
                    
                    self.connected = self.api.connect(
                        ip, 
                        port, 
                        time_out=self.config.tdx.timeout
                    )
                    
                    if self.connected:
                        self.current_server = (ip, port)
                        logger.info(f"✓ 连接成功: {ip}:{port}")
                        return True
                    else:
                        logger.debug(f"× 连接失败: {ip}:{port}")
                        
                except Exception as e:
                    logger.error(f"连接异常 {ip}:{port}: {str(e)}")
                    self.connected = False
            
            if attempt < self.config.tdx.retry_count - 1:
                wait_time = 2
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        logger.error("所有服务器连接均失败")
        return False
    
    def disconnect(self):
        """断开TDX服务器连接"""
        if self.connected:
            self.api.disconnect()
            self.connected = False
            logger.info(f"已断开服务器连接: {self.current_server}")
            self.current_server = None
    
    def get_stock_data(
        self, 
        stock_code: str, 
        market: int, 
        period: str = 'daily', 
        count: int = None
    ) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            market: 市场代码（0: 深圳, 1: 上海）
            period: 数据周期 ('daily' 或 '30min')
            count: 获取的数据条数
        
        Returns:
            DataFrame: 股票数据，包含 Open, High, Low, Close, Volume, Amount
        """
        if not self.connect():
            logger.error(f"无法连接服务器，获取数据失败: {stock_code}")
            return None
        
        # 使用配置中的默认值
        if count is None:
            count = self.config.data.daily_bars if period == 'daily' else self.config.data.min30_bars
        
        try:
            # 确定数据类型
            if period == 'daily':
                data_type = 9  # 日线数据
            elif period == '30min':
                data_type = 0  # 30分钟数据
            else:
                logger.error(f"不支持的周期: {period}")
                return None
            
            logger.debug(f"获取股票数据: {stock_code}, 周期: {period}, 数量: {count}")
            
            # 获取数据
            data = self.api.get_security_bars(
                data_type,
                market,
                stock_code,
                0,
                count
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning(f"获取到空数据: {stock_code}")
                return None
            
            # 统一列名
            df.rename(columns={
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'vol': 'Volume',
                'amount': 'Amount'
            }, inplace=True)
            
            # 处理日期
            df = self._parse_datetime(df, period)
            
            if df is None or df.empty:
                logger.warning(f"日期解析后数据为空: {stock_code}")
                return None
            
            # 设置日期索引
            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            logger.info(f"成功获取 {stock_code} 数据: {len(df)} 条记录")
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']]
        
        except Exception as e:
            logger.error(f"获取 {stock_code} 数据失败: {str(e)}", exc_info=True)
            return None
    
    def _parse_datetime(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        解析日期时间（改进的日期处理逻辑）
        
        Args:
            df: 原始DataFrame
            period: 数据周期
        
        Returns:
            处理后的DataFrame
        """
        try:
            if period == 'daily':
                # 日线数据：使用 year/month/day 组合
                df['Date'] = pd.to_datetime(
                    df[['year', 'month', 'day']]
                    .astype(str)
                    .apply(lambda x: f"{x['year']}-{x['month']}-{x['day']}", axis=1)
                )
            elif period == '30min':
                # 30分钟数据：灵活解析
                def parse_tdx_datetime(x):
                    try:
                        if isinstance(x, int):
                            s = str(x).zfill(12)
                            return datetime.datetime.strptime(s, "%Y%m%d%H%M")
                        elif '-' in str(x):
                            return pd.to_datetime(x, format='mixed')
                        else:
                            return pd.to_datetime(str(x), format='%Y%m%d%H%M', errors='coerce')
                    except:
                        return pd.NaT
                
                df['Date'] = df['datetime'].apply(parse_tdx_datetime)
                # 清理无效日期
                df = df[df['Date'].notna()]
            
            return df
        
        except Exception as e:
            logger.error(f"日期解析失败: {str(e)}", exc_info=True)
            return None
    
    def get_realtime_quotes(self, stock_list: list) -> pd.DataFrame:
        """
        获取实时行情（批量）
        
        Args:
            stock_list: 股票列表 [{'code': '600000', 'market': 1}, ...]
        
        Returns:
            DataFrame: 实时行情数据
        """
        if not self.connect():
            logger.error("无法连接服务器，获取实时行情失败")
            return None
        
        try:
            quotes = []
            for stock in stock_list:
                quote = self.api.get_security_quotes(
                    [(stock['market'], stock['code'])]
                )
                if quote:
                    quotes.extend(quote)
            
            df = pd.DataFrame(quotes)
            logger.info(f"成功获取 {len(quotes)} 只股票的实时行情")
            return df
        
        except Exception as e:
            logger.error(f"获取实时行情失败: {str(e)}", exc_info=True)
            return None
    
    def __enter__(self):
        """上下文管理器支持"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.disconnect()
    
    def __del__(self):
        """析构函数：确保连接关闭"""
        self.disconnect()
