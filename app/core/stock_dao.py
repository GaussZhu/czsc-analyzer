# -*- coding: utf-8 -*-
"""
Stock Data Access Object
股票数据访问对象 - 操作stock_kline数据库
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from app.core.db_connector import MySQLConnector
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class StockDAO:
    """股票数据访问对象"""
    
    def __init__(self, db_connector: MySQLConnector):
        """
        初始化DAO
        
        Args:
            db_connector: MySQL连接器
        """
        self.db = db_connector
        logger.info("StockDAO初始化完成")
    
    # ==================== code_map 操作 ====================
    
    def get_all_stock_codes(self) -> List[Dict]:
        """
        获取所有股票代码映射
        
        Returns:
            List[Dict]: 股票代码列表
        """
        query = "SELECT * FROM code_map"
        result = self.db.execute_query(query)
        return result if result else []
    
    def get_stock_info(self, code: str) -> Optional[Dict]:
        """
        获取股票信息
        
        Args:
            code: 股票代码
        
        Returns:
            Dict: 股票信息
        """
        query = "SELECT * FROM code_map WHERE code = %s"
        result = self.db.execute_query(query, (code,), fetch_one=True)
        return result
    
    def get_bj_stock_codes(self) -> List[Dict]:
        """
        获取北交所股票代码
        
        Returns:
            List[Dict]: 北交所股票列表
        """
        query = "SELECT * FROM bj_code_map"
        result = self.db.execute_query(query)
        return result if result else []
    
    # ==================== daily_kline 操作 ====================
    
    def get_daily_kline(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 300
    ) -> Optional[pd.DataFrame]:
        """
        获取日线K线数据
        
        Args:
            code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 限制条数
        
        Returns:
            DataFrame: K线数据
        """
        query = "SELECT * FROM daily_kline WHERE code = %s"
        params = [code]
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        # LIMIT不能使用参数化查询，必须直接插入数值
        query += f" ORDER BY date DESC LIMIT {limit}"
        
        df = self.db.query_to_dataframe(query, tuple(params))
        
        if df is not None and not df.empty:
            # 按日期升序排列（最早的在前）
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            logger.info(f"获取日线数据: {code}, {len(df)}条")
        
        return df
    
    def get_latest_daily_kline(self, code: str, days: int = 10) -> Optional[pd.DataFrame]:
        """
        获取最近N天的日线数据
        
        Args:
            code: 股票代码
            days: 天数
        
        Returns:
            DataFrame: K线数据
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')
        
        return self.get_daily_kline(code, start_date, end_date, limit=days)
    
    # ==================== 30min_kline 操作 ====================
    
    def get_30min_kline(
        self,
        code: str,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        limit: int = 300
    ) -> Optional[pd.DataFrame]:
        """
        获取30分钟K线数据
        
        Args:
            code: 股票代码
            start_datetime: 开始时间 (YYYY-MM-DD HH:MM:SS)
            end_datetime: 结束时间
            limit: 限制条数
        
        Returns:
            DataFrame: K线数据
        """
        query = "SELECT * FROM `30min_kline` WHERE code = %s"
        params = [code]
        
        if start_datetime:
            query += " AND datetime >= %s"
            params.append(start_datetime)
        
        if end_datetime:
            query += " AND datetime <= %s"
            params.append(end_datetime)
        
        # LIMIT不能使用参数化查询，必须直接插入数值
        query += f" ORDER BY datetime DESC LIMIT {limit}"
        
        df = self.db.query_to_dataframe(query, tuple(params))
        
        if df is not None and not df.empty:
            df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
            logger.info(f"获取30分钟数据: {code}, {len(df)}条")
        
        return df
    
    def get_latest_30min_kline(self, code: str, hours: int = 48) -> Optional[pd.DataFrame]:
        """
        获取最近N小时的30分钟数据
        
        Args:
            code: 股票代码
            hours: 小时数
        
        Returns:
            DataFrame: K线数据
        """
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_datetime = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        
        return self.get_30min_kline(code, start_datetime, end_datetime, limit=hours*2)
    
    # ==================== weekly_kline 操作 ====================
    
    def get_weekly_kline(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        获取周线K线数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制条数
        
        Returns:
            DataFrame: K线数据
        """
        query = "SELECT * FROM weekly_kline WHERE code = %s"
        params = [code]
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        # LIMIT不能使用参数化查询，必须直接插入数值
        query += f" ORDER BY date DESC LIMIT {limit}"
        
        df = self.db.query_to_dataframe(query, tuple(params))
        
        if df is not None and not df.empty:
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            logger.info(f"获取周线数据: {code}, {len(df)}条")
        
        return df
    
    # ==================== kronos 预测数据操作 ====================
    
    def get_kronos_predictions(self, code: str, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        获取Kronos预测数据
        
        Args:
            code: 股票代码
            limit: 限制条数
        
        Returns:
            DataFrame: 预测数据
        """
        # LIMIT不能使用参数化查询，必须直接插入数值
        query = f"SELECT * FROM kronos_pred WHERE code = %s ORDER BY date DESC LIMIT {limit}"
        df = self.db.query_to_dataframe(query, (code,))
        
        if df is not None and not df.empty:
            logger.info(f"获取Kronos预测: {code}, {len(df)}条")
        
        return df
    
    def get_kronos_eval_daily(self, code: str, limit: int = 30) -> Optional[pd.DataFrame]:
        """
        获取Kronos每日评估数据
        
        Args:
            code: 股票代码
            limit: 限制条数
        
        Returns:
            DataFrame: 评估数据
        """
        # LIMIT不能使用参数化查询，必须直接插入数值
        query = f"SELECT * FROM kronos_eval_daily WHERE code = %s ORDER BY date DESC LIMIT {limit}"
        df = self.db.query_to_dataframe(query, (code,))
        
        if df is not None and not df.empty:
            logger.info(f"获取Kronos评估: {code}, {len(df)}条")
        
        return df
    
    # ==================== 数据统计 ====================
    
    def get_data_range(self, table: str, code: str) -> Optional[Dict]:
        """
        获取某股票的数据时间范围
        
        Args:
            table: 表名
            code: 股票代码
        
        Returns:
            Dict: {'min_date': ..., 'max_date': ..., 'count': ...}
        """
        date_field = 'datetime' if table == '30min_kline' else 'date'
        
        query = f"""
            SELECT 
                MIN({date_field}) as min_date,
                MAX({date_field}) as max_date,
                COUNT(*) as count
            FROM `{table}`
            WHERE code = %s
        """
        
        result = self.db.execute_query(query, (code,), fetch_one=True)
        return result
    
    def get_all_stocks_summary(self) -> List[Dict]:
        """
        获取所有股票的数据概况
        
        Returns:
            List[Dict]: 股票概况列表
        """
        query = """
            SELECT 
                c.code,
                c.name,
                COUNT(DISTINCT d.date) as daily_count,
                MAX(d.date) as latest_date
            FROM code_map c
            LEFT JOIN daily_kline d ON c.code = d.code
            GROUP BY c.code, c.name
            ORDER BY latest_date DESC
        """
        
        result = self.db.execute_query(query)
        return result if result else []


# 便捷函数
def get_stock_dao(db_connector: MySQLConnector = None) -> StockDAO:
    """
    获取StockDAO实例（便捷函数）
    
    Args:
        db_connector: MySQL连接器，如果为None则创建默认连接
    
    Returns:
        StockDAO: DAO实例
    """
    if db_connector is None:
        from app.core.db_connector import get_mysql_connector
        db_connector = get_mysql_connector()
    
    return StockDAO(db_connector)
