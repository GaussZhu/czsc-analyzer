# -*- coding: utf-8 -*-
"""
Time Helper Module
时间工具模块
"""

import datetime
from app.config.settings import get_config
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class TradingTimeHelper:
    """交易时间辅助类"""
    
    def __init__(self):
        """初始化交易时间辅助类"""
        self.config = get_config()
        self.trading_hours = self.config.monitor.trading_hours
    
    def is_trading_day(self, date: datetime.date = None) -> bool:
        """
        判断是否为交易日（简化版，不考虑节假日）
        
        Args:
            date: 日期，默认为今天
        
        Returns:
            bool: 是否为交易日
        """
        if date is None:
            date = datetime.date.today()
        
        # 周末不是交易日
        if date.weekday() >= 5:
            logger.debug(f"{date} 是周末，非交易日")
            return False
        
        # TODO: 添加节假日判断
        return True
    
    def is_trading_hours(self, now: datetime.datetime = None) -> bool:
        """
        判断是否在交易时段内
        
        Args:
            now: 当前时间，默认为现在
        
        Returns:
            bool: 是否在交易时段
        """
        if now is None:
            now = datetime.datetime.now()
        
        current_time = now.time()
        
        # 上午时段
        morning_trading = (
            self.trading_hours.morning_start <= current_time <= self.trading_hours.morning_end
        )
        
        # 下午时段
        afternoon_trading = (
            self.trading_hours.afternoon_start <= current_time <= self.trading_hours.afternoon_end
        )
        
        return morning_trading or afternoon_trading
    
    def next_trading_time(self, now: datetime.datetime = None) -> datetime.datetime:
        """
        计算下一个交易时段的开始时间
        
        Args:
            now: 当前时间
        
        Returns:
            datetime: 下一个交易时段开始时间
        """
        if now is None:
            now = datetime.datetime.now()
        
        today = now.date()
        current_time = now.time()
        
        # 如果当前在上午交易时段前
        if current_time < self.trading_hours.morning_start:
            next_time = datetime.datetime.combine(today, self.trading_hours.morning_start)
        
        # 如果当前在上午和下午时段之间
        elif self.trading_hours.morning_end < current_time < self.trading_hours.afternoon_start:
            next_time = datetime.datetime.combine(today, self.trading_hours.afternoon_start)
        
        # 如果当前在下午交易时段后，则等待明天上午
        else:
            tomorrow = today + datetime.timedelta(days=1)
            # 跳过周末
            while tomorrow.weekday() >= 5:
                tomorrow += datetime.timedelta(days=1)
            next_time = datetime.datetime.combine(tomorrow, self.trading_hours.morning_start)
        
        return next_time
    
    def wait_until_next_trading(self):
        """等待到下一个交易时段"""
        import time
        
        while True:
            now = datetime.datetime.now()
            
            # 如果不是交易日，等待到下一个交易日
            if not self.is_trading_day(now.date()):
                next_time = self.next_trading_time(now)
                wait_seconds = (next_time - now).total_seconds()
                logger.info(f"⏳ 非交易日，下次检测时间：{next_time}")
                time.sleep(wait_seconds)
                continue
            
            # 如果不在交易时段，等待到下一个交易时段
            if not self.is_trading_hours(now):
                next_time = self.next_trading_time(now)
                wait_seconds = (next_time - now).total_seconds()
                logger.info(f"⏳ 非交易时段，下次检测时间：{next_time}")
                time.sleep(wait_seconds)
                continue
            
            # 已经是交易日且在交易时段内
            return
    
    def get_trading_status(self) -> dict:
        """
        获取当前交易状态
        
        Returns:
            dict: 交易状态信息
        """
        now = datetime.datetime.now()
        is_trading_day = self.is_trading_day(now.date())
        is_trading_hours = self.is_trading_hours(now)
        
        status = {
            'now': now,
            'is_trading_day': is_trading_day,
            'is_trading_hours': is_trading_hours,
            'can_trade': is_trading_day and is_trading_hours
        }
        
        if not status['can_trade']:
            status['next_trading_time'] = self.next_trading_time(now)
        
        return status


# 便捷函数
def is_trading_day(date: datetime.date = None) -> bool:
    """快速判断是否为交易日"""
    helper = TradingTimeHelper()
    return helper.is_trading_day(date)


def is_trading_hours(now: datetime.datetime = None) -> bool:
    """快速判断是否在交易时段"""
    helper = TradingTimeHelper()
    return helper.is_trading_hours(now)


def wait_until_next_trading():
    """快速等待到下一个交易时段"""
    helper = TradingTimeHelper()
    helper.wait_until_next_trading()
