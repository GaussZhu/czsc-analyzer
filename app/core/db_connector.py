# -*- coding: utf-8 -*-
"""
MySQL Database Connector Module
MySQL数据库连接模块
"""

import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
import pandas as pd
from typing import Optional, List, Dict, Any

from app.config.logging_config import get_logger

logger = get_logger(__name__)


class MySQLConnector:
    """MySQL数据库连接器"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 3306,
        user: str = 'root',
        password: str = '',
        database: str = 'stock_kline',
        charset: str = 'utf8mb4'
    ):
        """
        初始化MySQL连接器
        
        Args:
            host: 数据库主机
            port: 端口
            user: 用户名
            password: 密码
            database: 数据库名
            charset: 字符集
        """
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'charset': charset,
            'cursorclass': DictCursor
        }
        self.connection = None
        logger.info(f"MySQL连接器初始化: {host}:{port}/{database}")
    
    def connect(self) -> bool:
        """
        建立数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.connection = pymysql.connect(**self.config)
            logger.info("✓ MySQL连接成功")
            return True
        except Exception as e:
            logger.error(f"✗ MySQL连接失败: {str(e)}", exc_info=True)
            return False
    
    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("MySQL连接已关闭")
    
    @contextmanager
    def get_connection(self):
        """
        上下文管理器：获取数据库连接
        
        Yields:
            connection: 数据库连接对象
        """
        if not self.connection or not self.connection.open:
            self.connect()
        
        try:
            yield self.connection
        finally:
            pass  # 不在这里关闭连接，由disconnect方法统一管理
    
    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False
    ) -> Optional[Any]:
        """
        执行查询语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
            fetch_one: 是否只获取一条记录
        
        Returns:
            查询结果
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    
                    if fetch_one:
                        result = cursor.fetchone()
                    else:
                        result = cursor.fetchall()
                    
                    logger.debug(f"查询执行成功: {query[:100]}...")
                    return result
        
        except Exception as e:
            logger.error(f"查询执行失败: {str(e)}", exc_info=True)
            return None
    
    def execute_update(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        执行更新/插入/删除语句
        
        Args:
            query: SQL语句
            params: 参数
        
        Returns:
            int: 影响的行数
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    affected_rows = cursor.execute(query, params)
                    conn.commit()
                    
                    logger.debug(f"更新成功: 影响{affected_rows}行")
                    return affected_rows
        
        except Exception as e:
            logger.error(f"更新失败: {str(e)}", exc_info=True)
            if self.connection:
                self.connection.rollback()
            return 0
    
    def query_to_dataframe(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Optional[pd.DataFrame]:
        """
        查询结果转换为DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
        
        Returns:
            DataFrame: 查询结果
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
                logger.debug(f"查询到 {len(df)} 条记录")
                return df
        
        except Exception as e:
            logger.error(f"查询转DataFrame失败: {str(e)}", exc_info=True)
            return None
    
    def batch_insert(
        self,
        table: str,
        data: List[Dict],
        ignore: bool = False
    ) -> int:
        """
        批量插入数据
        
        Args:
            table: 表名
            data: 数据列表
            ignore: 是否使用INSERT IGNORE
        
        Returns:
            int: 插入的行数
        """
        if not data:
            return 0
        
        try:
            # 获取列名
            columns = list(data[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join([f"`{col}`" for col in columns])
            
            # 构建SQL
            insert_type = "INSERT IGNORE" if ignore else "INSERT"
            query = f"{insert_type} INTO `{table}` ({columns_str}) VALUES ({placeholders})"
            
            # 准备数据
            values = [tuple(row[col] for col in columns) for row in data]
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    affected_rows = cursor.executemany(query, values)
                    conn.commit()
                    
                    logger.info(f"批量插入成功: {table} 表，插入 {affected_rows} 行")
                    return affected_rows
        
        except Exception as e:
            logger.error(f"批量插入失败: {str(e)}", exc_info=True)
            if self.connection:
                self.connection.rollback()
            return 0
    
    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        
        Args:
            table_name: 表名
        
        Returns:
            bool: 表是否存在
        """
        query = """
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = %s
        """
        result = self.execute_query(query, (self.config['database'], table_name), fetch_one=True)
        return result and result['count'] > 0
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        获取表的列名
        
        Args:
            table_name: 表名
        
        Returns:
            List[str]: 列名列表
        """
        query = f"SHOW COLUMNS FROM `{table_name}`"
        result = self.execute_query(query)
        
        if result:
            return [row['Field'] for row in result]
        return []
    
    def __enter__(self):
        """上下文管理器支持"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.disconnect()
    
    def __del__(self):
        """析构函数"""
        self.disconnect()


# 便捷函数
def get_mysql_connector(
    host: str = 'localhost',
    port: int = 3306,
    user: str = 'root',
    password: str = '',
    database: str = 'stock_kline'
) -> MySQLConnector:
    """
    获取MySQL连接器（便捷函数）
    
    Args:
        host: 主机
        port: 端口
        user: 用户名
        password: 密码
        database: 数据库名
    
    Returns:
        MySQLConnector: 连接器实例
    """
    return MySQLConnector(host, port, user, password, database)
