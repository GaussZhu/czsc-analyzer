# -*- coding: utf-8 -*-
"""
Technical Indicators Module
技术指标计算模块
"""

import pandas as pd
import numpy as np
from app.config.settings import get_config
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self):
        """初始化技术指标计算器"""
        self.config = get_config()
        logger.debug("技术指标计算器初始化完成")
    
    def calculate_macd(
        self, 
        data: pd.DataFrame, 
        fast: int = None, 
        slow: int = None, 
        signal: int = None
    ) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: DataFrame，必须包含 'Close' 列
            fast: 快速EMA周期
            slow: 慢速EMA周期
            signal: 信号线周期
        
        Returns:
            DataFrame: 添加了MACD相关列的数据
        """
        # 使用配置的默认值
        fast = fast or self.config.macd.fast
        slow = slow or self.config.macd.slow
        signal = signal or self.config.macd.signal
        
        logger.debug(f"计算MACD: fast={fast}, slow={slow}, signal={signal}")
        
        df = data.copy()
        
        try:
            # 计算快速和慢速EMA
            df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
            df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
            
            # 计算MACD线
            df['MACD'] = df['EMA_fast'] - df['EMA_slow']
            
            # 计算信号线
            df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            
            # 计算柱状图
            df['Histogram'] = df['MACD'] - df['Signal']
            
            logger.debug(f"MACD计算完成，数据行数: {len(df)}")
            return df
        
        except Exception as e:
            logger.error(f"MACD计算失败: {str(e)}", exc_info=True)
            return data
    
    def calculate_ma(self, data: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """
        计算移动平均线
        
        Args:
            data: DataFrame，必须包含 'Close' 列
            periods: 周期列表，例如 [5, 10, 20, 60]
        
        Returns:
            DataFrame: 添加了MA列的数据
        """
        if periods is None:
            periods = [5, 10, 20, 60]
        
        df = data.copy()
        
        try:
            for period in periods:
                df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
            
            logger.debug(f"移动平均线计算完成: {periods}")
            return df
        
        except Exception as e:
            logger.error(f"移动平均线计算失败: {str(e)}", exc_info=True)
            return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            data: DataFrame，必须包含 'Close' 列
            period: RSI周期
        
        Returns:
            DataFrame: 添加了RSI列的数据
        """
        df = data.copy()
        
        try:
            # 计算价格变化
            delta = df['Close'].diff()
            
            # 分离涨跌
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # 计算RS和RSI
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            logger.debug(f"RSI计算完成，周期: {period}")
            return df
        
        except Exception as e:
            logger.error(f"RSI计算失败: {str(e)}", exc_info=True)
            return data
    
    def calculate_bollinger_bands(
        self, 
        data: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        计算布林带
        
        Args:
            data: DataFrame，必须包含 'Close' 列
            period: 移动平均周期
            std_dev: 标准差倍数
        
        Returns:
            DataFrame: 添加了布林带列的数据
        """
        df = data.copy()
        
        try:
            # 计算中轨（移动平均）
            df['BB_middle'] = df['Close'].rolling(window=period).mean()
            
            # 计算标准差
            rolling_std = df['Close'].rolling(window=period).std()
            
            # 计算上下轨
            df['BB_upper'] = df['BB_middle'] + (rolling_std * std_dev)
            df['BB_lower'] = df['BB_middle'] - (rolling_std * std_dev)
            
            logger.debug(f"布林带计算完成: period={period}, std_dev={std_dev}")
            return df
        
        except Exception as e:
            logger.error(f"布林带计算失败: {str(e)}", exc_info=True)
            return data
    
    def calculate_kdj(self, data: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            data: DataFrame，必须包含 'High', 'Low', 'Close' 列
            n: RSV周期
            m1: K值平滑周期
            m2: D值平滑周期
        
        Returns:
            DataFrame: 添加了KDJ列的数据
        """
        df = data.copy()
        
        try:
            # 计算RSV
            low_list = df['Low'].rolling(window=n).min()
            high_list = df['High'].rolling(window=n).max()
            rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
            
            # 计算K、D、J值
            df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
            df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            
            logger.debug(f"KDJ计算完成: n={n}, m1={m1}, m2={m2}")
            return df
        
        except Exception as e:
            logger.error(f"KDJ计算失败: {str(e)}", exc_info=True)
            return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算ATR（平均真实波幅）
        
        Args:
            data: DataFrame，必须包含 'High', 'Low', 'Close' 列
            period: ATR周期
        
        Returns:
            DataFrame: 添加了ATR列的数据
        """
        df = data.copy()
        
        try:
            # 计算真实波幅（TR）
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
            df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
            
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            
            # 计算ATR
            df['ATR'] = df['TR'].rolling(window=period).mean()
            
            # 清理临时列
            df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
            
            logger.debug(f"ATR计算完成，周期: {period}")
            return df
        
        except Exception as e:
            logger.error(f"ATR计算失败: {str(e)}", exc_info=True)
            return data
    
    def calculate_volume_ma(self, data: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """
        计算成交量移动平均
        
        Args:
            data: DataFrame，必须包含 'Volume' 列
            periods: 周期列表
        
        Returns:
            DataFrame: 添加了成交量MA列的数据
        """
        if periods is None:
            periods = [5, 10, 20]
        
        df = data.copy()
        
        try:
            for period in periods:
                df[f'VolumeMA{period}'] = df['Volume'].rolling(window=period).mean()
            
            logger.debug(f"成交量移动平均计算完成: {periods}")
            return df
        
        except Exception as e:
            logger.error(f"成交量移动平均计算失败: {str(e)}", exc_info=True)
            return data
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加所有常用技术指标
        
        Args:
            data: 原始数据
        
        Returns:
            DataFrame: 包含所有指标的数据
        """
        logger.info("开始计算所有技术指标")
        
        df = data.copy()
        df = self.calculate_macd(df)
        df = self.calculate_ma(df)
        df = self.calculate_rsi(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_kdj(df)
        df = self.calculate_atr(df)
        df = self.calculate_volume_ma(df)
        
        logger.info("所有技术指标计算完成")
        return df


# 便捷函数
def calculate_macd(data: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """快速计算MACD"""
    calculator = TechnicalIndicators()
    return calculator.calculate_macd(data, fast, slow, signal)


def calculate_ma(data: pd.DataFrame, periods=None) -> pd.DataFrame:
    """快速计算移动平均线"""
    calculator = TechnicalIndicators()
    return calculator.calculate_ma(data, periods)
