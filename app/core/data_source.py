# -*- coding: utf-8 -*-
"""
Unified Data Source Module
统一数据源模块 - 支持TDX和MySQL两种数据源
"""

import pandas as pd
from typing import Optional
from enum import Enum

from app.core.data_fetcher import TDXDataFetcher
from app.core.db_connector import MySQLConnector
from app.core.stock_dao import StockDAO
from app.config.settings import get_config
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class DataSourceType(Enum):
    """数据源类型"""
    TDX = "tdx"          # 通达信实时数据
    MYSQL = "mysql"      # MySQL历史数据
    AUTO = "auto"        # 自动选择


class UnifiedDataSource:
    """统一数据源"""
    
    def __init__(self, source_type: DataSourceType = DataSourceType.AUTO):
        """
        初始化统一数据源
        
        Args:
            source_type: 数据源类型
        """
        self.config = get_config()
        self.source_type = source_type
        
        # 初始化TDX数据获取器
        self.tdx_fetcher = TDXDataFetcher()
        
        # 初始化MySQL连接（如果配置了）
        self.mysql_connector = None
        self.stock_dao = None
        
        if self.config.mysql.use_mysql_data:
            try:
                self.mysql_connector = MySQLConnector(
                    host=self.config.mysql.host,
                    port=self.config.mysql.port,
                    user=self.config.mysql.user,
                    password=self.config.mysql.password,
                    database=self.config.mysql.database
                )
                self.stock_dao = StockDAO(self.mysql_connector)
                logger.info("MySQL数据源初始化成功")
            except Exception as e:
                logger.warning(f"MySQL数据源初始化失败: {str(e)}")
        
        logger.info(f"统一数据源初始化完成，类型: {source_type.value}")
    
    def get_stock_data(
        self,
        stock_code: str,
        market: int,
        period: str = 'daily',
        count: int = 300,
        prefer_mysql: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取股票数据（统一接口）
        
        Args:
            stock_code: 股票代码
            market: 市场代码
            period: 数据周期
            count: 数据条数
            prefer_mysql: 优先使用MySQL数据
        
        Returns:
            DataFrame: 股票数据
        """
        data = None
        
        # 决定使用哪个数据源
        use_mysql = (
            prefer_mysql and 
            self.stock_dao is not None and 
            self.source_type in [DataSourceType.MYSQL, DataSourceType.AUTO]
        )
        
        # 先尝试MySQL
        if use_mysql:
            data = self._get_data_from_mysql(stock_code, period, count)
            
            if data is not None:
                logger.info(f"从MySQL获取数据: {stock_code}, {len(data)}条")
                return self._normalize_dataframe(data, period)
        
        # 如果MySQL失败或未配置，使用TDX
        if data is None and self.source_type != DataSourceType.MYSQL:
            logger.info(f"从TDX获取数据: {stock_code}")
            data = self.tdx_fetcher.get_stock_data(stock_code, market, period, count)
        
        return data
    
    def _get_data_from_mysql(
        self,
        stock_code: str,
        period: str,
        count: int
    ) -> Optional[pd.DataFrame]:
        """
        从MySQL获取数据
        
        Args:
            stock_code: 股票代码
            period: 周期
            count: 数量
        
        Returns:
            DataFrame: 数据
        """
        try:
            if period == 'daily':
                df = self.stock_dao.get_daily_kline(stock_code, limit=count)
            elif period == '30min':
                df = self.stock_dao.get_30min_kline(stock_code, limit=count)
            elif period == 'weekly':
                df = self.stock_dao.get_weekly_kline(stock_code, limit=count)
            else:
                logger.warning(f"不支持的周期: {period}")
                return None
            
            return df
        
        except Exception as e:
            logger.error(f"从MySQL获取数据失败: {str(e)}", exc_info=True)
            return None
    
    def _normalize_dataframe(
        self,
        df: pd.DataFrame,
        period: str
    ) -> pd.DataFrame:
        """
        标准化DataFrame格式（统一MySQL和TDX的数据格式）
        
        Args:
            df: 原始DataFrame
            period: 周期
        
        Returns:
            DataFrame: 标准化后的DataFrame
        """
        if df is None or df.empty:
            return df
        
        # 复制数据
        result = df.copy()
        
        # 标准化列名（统一为大写开头）
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'vol': 'Volume',
            'amount': 'Amount'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in result.columns:
                result.rename(columns={old_col: new_col}, inplace=True)
        
        # 设置索引
        if period == 'daily' or period == 'weekly':
            if 'date' in result.columns:
                result['Date'] = pd.to_datetime(result['date'])
                result.set_index('Date', inplace=True)
        elif period == '30min':
            if 'datetime' in result.columns:
                result['Date'] = pd.to_datetime(result['datetime'])
                result.set_index('Date', inplace=True)
        
        # 确保有必需的列
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in result.columns:
                logger.warning(f"缺少必需列: {col}")
        
        # 只返回必需的列（如果存在）
        available_cols = [col for col in required_cols if col in result.columns]
        if 'Amount' in result.columns:
            available_cols.append('Amount')
        
        result = result[available_cols]
        
        # 按日期排序
        result.sort_index(ascending=True, inplace=True)
        
        return result
    
    def get_stock_list_from_db(self) -> Optional[list]:
        """
        从数据库获取股票列表
        
        Returns:
            list: 股票列表
        """
        if self.stock_dao is None:
            return None
        
        try:
            stocks = self.stock_dao.get_all_stock_codes()
            
            # 转换为标准格式
            result = []
            for stock in stocks:
                result.append({
                    'code': stock.get('code'),
                    'name': stock.get('name', ''),
                    'market': stock.get('market', 1)  # 默认上海市场
                })
            
            logger.info(f"从数据库获取 {len(result)} 只股票")
            return result
        
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}", exc_info=True)
            return None
    
    def save_analysis_results(self, results: dict):
        """
        保存分析结果到数据库
        
        Args:
            results: 分析结果
        """
        # TODO: 实现保存分析结果的逻辑
        # 可以创建新表来存储CZSC分析结果
        logger.info("保存分析结果（功能待实现）")
    
    def disconnect(self):
        """断开所有连接"""
        if self.tdx_fetcher:
            self.tdx_fetcher.disconnect()
        
        if self.mysql_connector:
            self.mysql_connector.disconnect()
        
        logger.info("所有数据源连接已关闭")
    
    def __enter__(self):
        """上下文管理器支持"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.disconnect()
    
    def __del__(self):
        """析构函数"""
        self.disconnect()


# 便捷函数
def get_unified_data_source(
    source_type: DataSourceType = DataSourceType.AUTO
) -> UnifiedDataSource:
    """
    获取统一数据源（便捷函数）
    
    Args:
        source_type: 数据源类型
    
    Returns:
        UnifiedDataSource: 统一数据源实例
    """
    return UnifiedDataSource(source_type)
