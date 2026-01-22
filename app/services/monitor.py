# -*- coding: utf-8 -*-
"""
Monitor Service Module
监控服务 - 自动监控股票并发送通知
"""

import time
import datetime
from typing import List, Dict

from app.services.analyzer import StockAnalyzer
from app.services.notifier import (
    NotificationManager, 
    EmailNotifier, 
    WeChatWorkNotifier, 
    DingTalkNotifier,
    NotificationLevel
)
from app.utils.time_helper import TradingTimeHelper
from app.config.settings import get_config
from app.config.logging_config import get_monitor_logger

logger = get_monitor_logger()


class StockMonitor:
    """股票监控服务"""
    
    def __init__(self, stock_list: List[Dict] = None):
        """
        初始化股票监控服务
        
        Args:
            stock_list: 要监控的股票列表
        """
        self.config = get_config()
        self.stock_list = stock_list or self.config.stock_list.stocks
        self.analyzer = StockAnalyzer()
        self.time_helper = TradingTimeHelper()
        
        # 初始化通知管理器
        self.notification_manager = NotificationManager()
        self._setup_notifiers()
        
        logger.info(f"股票监控服务初始化完成，监控 {len(self.stock_list)} 只股票")
    
    def _setup_notifiers(self):
        """设置通知器"""
        if not self.config.notification.enabled:
            logger.info("通知功能未启用")
            return
        
        # 邮件通知
        if self.config.notification.email_enabled and self.config.notification.email_from:
            try:
                email_notifier = EmailNotifier(
                    smtp_server=self.config.notification.smtp_server,
                    smtp_port=self.config.notification.smtp_port,
                    username=self.config.notification.email_username,
                    password=self.config.notification.email_password,
                    from_addr=self.config.notification.email_from,
                    to_addrs=self.config.notification.email_to,
                    enabled=True
                )
                self.notification_manager.add_notifier(email_notifier)
                logger.info("✓ 邮件通知已启用")
            except Exception as e:
                logger.warning(f"邮件通知初始化失败: {str(e)}")
        
        # 企业微信通知
        if self.config.notification.wechat_enabled and self.config.notification.wechat_webhook:
            try:
                wechat_notifier = WeChatWorkNotifier(
                    webhook_url=self.config.notification.wechat_webhook,
                    enabled=True
                )
                self.notification_manager.add_notifier(wechat_notifier)
                logger.info("✓ 企业微信通知已启用")
            except Exception as e:
                logger.warning(f"企业微信通知初始化失败: {str(e)}")
        
        # 钉钉通知
        if self.config.notification.dingtalk_enabled and self.config.notification.dingtalk_webhook:
            try:
                dingtalk_notifier = DingTalkNotifier(
                    webhook_url=self.config.notification.dingtalk_webhook,
                    secret=self.config.notification.dingtalk_secret if self.config.notification.dingtalk_secret else None,
                    enabled=True
                )
                self.notification_manager.add_notifier(dingtalk_notifier)
                logger.info("✓ 钉钉通知已启用")
            except Exception as e:
                logger.warning(f"钉钉通知初始化失败: {str(e)}")
    
    def monitor_stocks(self, interval: int = None):
        """
        持续监控股票（仅在交易时段运行）
        
        Args:
            interval: 监控间隔（秒），默认使用配置值
        """
        interval = interval or self.config.monitor.interval
        
        logger.info(f"开始监控，刷新间隔: {interval}秒")
        print(f"\n{'='*80}")
        print(f"CZSC股票分析工具 - 监控模式")
        print(f"监控股票数量: {len(self.stock_list)}")
        print(f"刷新间隔: {interval}秒")
        print(f"{'='*80}\n")
        
        try:
            while True:
                # 检查是否为交易日和交易时段
                if not self._should_monitor():
                    self.time_helper.wait_until_next_trading()
                    continue
                
                # 执行监控
                self._run_monitoring_cycle()
                
                # 等待下一次监控
                logger.info(f"⏳ 下次更新将在 {interval} 秒后...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("监控被用户中断")
            print("\n\n✓ 监控已停止")
        except Exception as e:
            logger.error(f"监控过程中发生错误: {str(e)}", exc_info=True)
            print(f"\n✗ 监控错误: {str(e)}")
        finally:
            self.cleanup()
    
    def _should_monitor(self) -> bool:
        """
        判断当前是否应该监控
        
        Returns:
            bool: 是否应该监控
        """
        status = self.time_helper.get_trading_status()
        
        if not status['is_trading_day']:
            logger.info("💤 当前为非交易日")
            return False
        
        if not status['is_trading_hours']:
            logger.info("💤 当前为非交易时段")
            return False
        
        return True
    
    def _run_monitoring_cycle(self):
        """执行一次监控周期"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 开始监控周期 [{datetime.datetime.now()}]")
        logger.info(f"{'='*60}")
        
        print(f"\n\n{'='*80}")
        print(f"刷新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('-'*80)
        
        for stock in self.stock_list:
            self._monitor_single_stock(stock)
    
    def _monitor_single_stock(self, stock: Dict):
        """
        监控单只股票
        
        Args:
            stock: 股票信息
        """
        logger.info(f"\n分析股票: {stock['name']} ({stock['code']})")
        
        # 日线分析
        daily_result = self.analyzer.analyze_stock(
            stock['code'],
            stock['name'],
            stock['market'],
            'daily'
        )
        
        if daily_result:
            self._check_and_alert(daily_result, '日线')
        
        # 30分钟线分析
        min30_result = self.analyzer.analyze_stock(
            stock['code'],
            stock['name'],
            stock['market'],
            '30min'
        )
        
        if min30_result:
            self._check_and_alert(min30_result, '30分钟')
    
    def _check_and_alert(self, result: Dict, period_name: str):
        """
        检查并发送警报
        
        Args:
            result: 分析结果
            period_name: 周期名称
        """
        signals = result['signals']
        has_signal = False
        
        # 检查买点
        for signal_type, points in signals['buy'].items():
            if points:
                has_signal = True
                for point in points:
                    self._send_alert(
                        result['stock'],
                        '买入',
                        signal_type,
                        point,
                        period_name
                    )
        
        # 检查卖点
        for signal_type, points in signals['sell'].items():
            if points:
                has_signal = True
                for point in points:
                    self._send_alert(
                        result['stock'],
                        '卖出',
                        signal_type,
                        point,
                        period_name
                    )
        
        # 如果有信号，显示详细结果
        if has_signal:
            print()  # 换行
            self.analyzer.print_analysis_results(result, max_display=3)
    
    def _send_alert(
        self,
        stock: Dict,
        action: str,
        signal_type: str,
        point: Dict,
        period_name: str
    ):
        """
        发送警报
        
        Args:
            stock: 股票信息
            action: 操作类型（买入/卖出）
            signal_type: 信号类型
            point: 信号点信息
            period_name: 周期名称
        """
        time_str = point['time'].strftime('%Y-%m-%d %H:%M')
        
        alert_msg = (
            f"🔔 【{action}信号】{stock['name']}({stock['code']}) - {period_name}\n"
            f"   类型: {signal_type}\n"
            f"   时间: {time_str}\n"
            f"   价格: {point['price']:.2f}\n"
            f"   原因: {point.get('reason', '未说明')}"
        )
        
        print(alert_msg)
        logger.warning(alert_msg)
        
        # 使用通知管理器发送通知
        if self.config.notification.enabled:
            signal_info = {
                'type': f"{action}_{signal_type}",
                'time': point['time'],
                'price': point['price'],
                'reason': point.get('reason', '未说明')
            }
            
            self.notification_manager.send_signal_alert(stock, signal_info, period_name)
    
    def scan_once(self) -> List[Dict]:
        """
        扫描一次所有股票（不循环）
        
        Returns:
            List[Dict]: 分析结果列表
        """
        logger.info("执行一次性扫描")
        print(f"\n{'='*80}")
        print(f"CZSC股票分析工具 - 扫描模式")
        print(f"扫描时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        results = []
        
        for stock in self.stock_list:
            print(f"\n分析 {stock['name']} ({stock['code']})...")
            
            # 日线分析
            daily_result = self.analyzer.analyze_stock(
                stock['code'],
                stock['name'],
                stock['market'],
                'daily'
            )
            
            if daily_result:
                results.append(daily_result)
                self.analyzer.print_analysis_results(daily_result)
        
        return results
    
    def get_signals_overview(self) -> List[Dict]:
        """
        获取所有股票的信号概览
        
        Returns:
            List[Dict]: 信号概览列表
        """
        overview = []
        
        for stock in self.stock_list:
            daily_result = self.analyzer.analyze_stock(
                stock['code'],
                stock['name'],
                stock['market'],
                'daily'
            )
            
            if daily_result:
                summary = self.analyzer.get_signals_summary(daily_result)
                overview.append(summary)
        
        return overview
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理监控资源")
        if hasattr(self, 'analyzer'):
            del self.analyzer


# 便捷函数
def start_monitoring(stock_list: List[Dict] = None, interval: int = None):
    """快速启动监控"""
    monitor = StockMonitor(stock_list)
    monitor.monitor_stocks(interval)


def scan_once(stock_list: List[Dict] = None) -> List[Dict]:
    """快速执行一次扫描"""
    monitor = StockMonitor(stock_list)
    return monitor.scan_once()
