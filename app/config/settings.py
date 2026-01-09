# -*- coding: utf-8 -*-
"""
Configuration Management Module
统一管理所有配置参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import time
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class TDXConfig:
    """通达信服务器配置"""
    servers: List[Tuple[str, int]] = field(default_factory=lambda: [
        ('sztdx.gtjas.com', 7709),
        ('60.12.136.250', 7709),
        ('60.191.117.167', 7709),
        ('218.75.126.9', 7709),
        ('115.238.56.198', 7709),
        ('115.238.90.165', 7709)
    ])
    timeout: int = 3
    retry_count: int = 3


@dataclass
class TradingHoursConfig:
    """交易时段配置"""
    morning_start: time = time(9, 25)
    morning_end: time = time(11, 30)
    afternoon_start: time = time(12, 55)
    afternoon_end: time = time(15, 5)


@dataclass
class MonitorConfig:
    """监控配置"""
    interval: int = 300  # 监控间隔（秒）
    recent_days: int = 3  # 日线信号：最近几天
    recent_hours: int = 6  # 分钟线信号：最近几小时
    trading_hours: TradingHoursConfig = field(default_factory=TradingHoursConfig)


@dataclass
class MACDConfig:
    """MACD指标参数配置"""
    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass
class FractalConfig:
    """分型识别配置"""
    n: int = 2  # 左右K线数量


@dataclass
class DataConfig:
    """数据配置"""
    daily_bars: int = 300  # 日线数据条数
    min30_bars: int = 300  # 30分钟数据条数
    cache_dir: str = "data/cache"
    db_dir: str = "data/db"
    log_dir: str = "data/logs"


@dataclass
class MySQLConfig:
    """MySQL数据库配置"""
    host: str = os.getenv('MYSQL_HOST', 'localhost')
    port: int = int(os.getenv('MYSQL_PORT', '3306'))
    user: str = os.getenv('MYSQL_USER', 'root')
    password: str = os.getenv('MYSQL_PASSWORD', '')
    database: str = os.getenv('MYSQL_DATABASE', 'stock_kline')
    charset: str = 'utf8mb4'
    
    # Unix socket路径（如果MYSQL_HOST以/开头或包含.sock，则视为socket路径）
    @property
    def unix_socket(self) -> Optional[str]:
        """判断是否使用Unix socket连接"""
        if self.host.startswith('/') or '.sock' in self.host:
            return self.host
        return os.getenv('MYSQL_UNIX_SOCKET', '')  # 默认不使用socket
    
    # 是否使用数据库作为数据源
    use_mysql_data: bool = os.getenv('USE_MYSQL_DATA', 'True').lower() == 'true'


@dataclass
class NotificationConfig:
    """通知配置"""
    # 是否启用通知
    enabled: bool = os.getenv('NOTIFICATION_ENABLED', 'True').lower() == 'true'
    
    # 邮件通知配置
    email_enabled: bool = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
    smtp_server: str = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port: int = int(os.getenv('SMTP_PORT', '587'))
    email_username: str = os.getenv('EMAIL_USERNAME', '')
    email_password: str = os.getenv('EMAIL_PASSWORD', '')
    email_from: str = os.getenv('EMAIL_FROM', '')
    email_to: List[str] = field(default_factory=lambda: 
        os.getenv('EMAIL_TO', '').split(',') if os.getenv('EMAIL_TO') else []
    )
    
    # 企业微信通知配置
    wechat_enabled: bool = os.getenv('WECHAT_ENABLED', 'False').lower() == 'true'
    wechat_webhook: str = os.getenv('WECHAT_WEBHOOK', '')
    
    # 钉钉通知配置
    dingtalk_enabled: bool = os.getenv('DINGTALK_ENABLED', 'False').lower() == 'true'
    dingtalk_webhook: str = os.getenv('DINGTALK_WEBHOOK', '')
    dingtalk_secret: str = os.getenv('DINGTALK_SECRET', '')


@dataclass
class StockListConfig:
    """股票列表配置"""
    # 默认监控的股票列表
    stocks: List[Dict] = field(default_factory=lambda: [
        {'code': '600000', 'name': '浦发银行', 'market': 1},  # TDXParams.MARKET_SH
        {'code': '600004', 'name': '白云机场', 'market': 1},
        {'code': '600006', 'name': '东风股份', 'market': 1},
    ])


@dataclass
class AppConfig:
    """应用总配置"""
    # 基础路径
    base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 各模块配置
    tdx: TDXConfig = field(default_factory=TDXConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    macd: MACDConfig = field(default_factory=MACDConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    stock_list: StockListConfig = field(default_factory=StockListConfig)
    
    # 调试模式
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    def __post_init__(self):
        """初始化后处理：创建必要的目录"""
        for dir_path in [self.data.cache_dir, self.data.db_dir, self.data.log_dir]:
            full_path = os.path.join(self.base_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)


# 全局配置实例
config = AppConfig()


def get_config() -> AppConfig:
    """获取全局配置实例"""
    return config


def update_stock_list(stocks: List[Dict]):
    """更新监控股票列表"""
    config.stock_list.stocks = stocks


# 环境变量支持
def load_from_env():
    """从环境变量加载配置"""
    if os.getenv('MONITOR_INTERVAL'):
        config.monitor.interval = int(os.getenv('MONITOR_INTERVAL'))
    
    if os.getenv('TDX_TIMEOUT'):
        config.tdx.timeout = int(os.getenv('TDX_TIMEOUT'))
    
    if os.getenv('MACD_FAST'):
        config.macd.fast = int(os.getenv('MACD_FAST'))
    
    if os.getenv('MACD_SLOW'):
        config.macd.slow = int(os.getenv('MACD_SLOW'))
    
    if os.getenv('MACD_SIGNAL'):
        config.macd.signal = int(os.getenv('MACD_SIGNAL'))


# 自动加载环境变量
load_from_env()
