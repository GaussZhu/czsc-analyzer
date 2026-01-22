# -*- coding: utf-8 -*-
"""
Logging Configuration Module
统一的日志管理系统
"""

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime


class LoggerConfig:
    """日志配置类"""
    
    def __init__(self, log_dir="data/logs", debug=False):
        self.log_dir = log_dir
        self.debug = debug
        self.log_level = logging.DEBUG if debug else logging.INFO
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志格式
        self.formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 详细格式（用于调试）
        self.detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取配置好的日志记录器
        
        Args:
            name: 日志记录器名称（通常使用 __name__）
        
        Returns:
            配置好的 Logger 实例
        """
        logger = logging.getLogger(name)
        
        # 避免重复添加 handler
        if logger.handlers:
            return logger
        
        logger.setLevel(self.log_level)
        
        # 1. 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)
        
        # 2. 通用日志文件（按大小轮转）
        general_log = os.path.join(self.log_dir, 'app.log')
        file_handler = RotatingFileHandler(
            general_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.detailed_formatter if self.debug else self.formatter)
        logger.addHandler(file_handler)
        
        # 3. 错误日志文件
        error_log = os.path.join(self.log_dir, 'error.log')
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.detailed_formatter)
        logger.addHandler(error_handler)
        
        # 4. 按日期轮转的日志（可选）
        daily_log = os.path.join(self.log_dir, f'daily_{datetime.now().strftime("%Y%m%d")}.log')
        daily_handler = TimedRotatingFileHandler(
            daily_log,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        daily_handler.setLevel(self.log_level)
        daily_handler.setFormatter(self.formatter)
        logger.addHandler(daily_handler)
        
        return logger


# 全局日志配置实例
_logger_config = None


def setup_logging(log_dir="data/logs", debug=False):
    """
    初始化全局日志配置
    
    Args:
        log_dir: 日志目录
        debug: 是否开启调试模式
    """
    global _logger_config
    _logger_config = LoggerConfig(log_dir, debug)


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器（便捷函数）
    
    Args:
        name: 日志记录器名称
    
    Returns:
        Logger 实例
    """
    global _logger_config
    if _logger_config is None:
        setup_logging()
    return _logger_config.get_logger(name)


# 特定模块的日志记录器
def get_data_logger() -> logging.Logger:
    """获取数据模块的日志记录器"""
    return get_logger('czsc.data')


def get_analysis_logger() -> logging.Logger:
    """获取分析模块的日志记录器"""
    return get_logger('czsc.analysis')


def get_monitor_logger() -> logging.Logger:
    """获取监控模块的日志记录器"""
    return get_logger('czsc.monitor')


def get_api_logger() -> logging.Logger:
    """获取API模块的日志记录器"""
    return get_logger('czsc.api')


# 日志装饰器
def log_execution(logger=None):
    """
    函数执行日志装饰器
    
    Args:
        logger: 日志记录器，如果为None则使用默认logger
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            logger.debug(f"执行函数: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数 {func.__name__} 执行成功")
                return result
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行失败: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


# 初始化默认日志配置
setup_logging()
