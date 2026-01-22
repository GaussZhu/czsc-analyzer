"""
API依赖注入模块
提供常用的依赖项，如数据库连接、服务实例等
"""
from typing import Optional
from app.core.data_source import UnifiedDataSource
from app.services.analyzer import StockAnalyzer
from app.config.settings import get_config

def get_data_source() -> UnifiedDataSource:
    """
    获取数据源实例
    每次请求创建新实例，确保数据实时性
    """
    return UnifiedDataSource()

def get_analyzer() -> StockAnalyzer:
    """
    获取股票分析器实例
    每次请求创建新实例，确保数据实时性
    """
    return StockAnalyzer()

def get_app_config():
    """获取应用配置"""
    return get_config()
