# -*- coding: utf-8 -*-
"""
Notification Service Module
通知服务 - 支持邮件、企业微信、钉钉等多种通知方式
"""

import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from enum import Enum

from app.config.logging_config import get_logger

logger = get_logger(__name__)


class NotificationType(Enum):
    """通知类型"""
    EMAIL = "email"
    WECHAT_WORK = "wechat_work"  # 企业微信
    DINGTALK = "dingtalk"        # 钉钉
    CONSOLE = "console"           # 控制台输出


class NotificationLevel(Enum):
    """通知级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BaseNotifier:
    """通知器基类"""
    
    def __init__(self, enabled: bool = True):
        """
        初始化通知器
        
        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
    
    def send(self, title: str, content: str, level: NotificationLevel = NotificationLevel.INFO):
        """
        发送通知
        
        Args:
            title: 标题
            content: 内容
            level: 通知级别
        """
        if not self.enabled:
            logger.debug("通知器未启用，跳过发送")
            return
        
        try:
            self._do_send(title, content, level)
        except Exception as e:
            logger.error(f"发送通知失败: {str(e)}", exc_info=True)
    
    def _do_send(self, title: str, content: str, level: NotificationLevel):
        """子类实现具体的发送逻辑"""
        raise NotImplementedError


class EmailNotifier(BaseNotifier):
    """邮件通知器"""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True,
        enabled: bool = True
    ):
        """
        初始化邮件通知器
        
        Args:
            smtp_server: SMTP服务器地址
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            from_addr: 发件人地址
            to_addrs: 收件人地址列表
            use_tls: 是否使用TLS
            enabled: 是否启用
        """
        super().__init__(enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
        
        logger.info(f"邮件通知器初始化: {smtp_server}:{smtp_port}")
    
    def _do_send(self, title: str, content: str, level: NotificationLevel):
        """发送邮件"""
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.to_addrs)
        msg['Subject'] = f"[{level.value.upper()}] {title}"
        
        # 添加内容
        body = MIMEText(content, 'plain', 'utf-8')
        msg.attach(body)
        
        # 发送邮件
        try:
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            
            server.login(self.username, self.password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            server.quit()
            
            logger.info(f"✓ 邮件发送成功: {title}")
        
        except Exception as e:
            logger.error(f"✗ 邮件发送失败: {str(e)}")
            raise


class WeChatWorkNotifier(BaseNotifier):
    """企业微信通知器"""
    
    def __init__(self, webhook_url: str, enabled: bool = True):
        """
        初始化企业微信通知器
        
        Args:
            webhook_url: 企业微信机器人Webhook URL
            enabled: 是否启用
        """
        super().__init__(enabled)
        self.webhook_url = webhook_url
        logger.info("企业微信通知器初始化")
    
    def _do_send(self, title: str, content: str, level: NotificationLevel):
        """发送企业微信消息"""
        # 构建消息
        emoji_map = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🔴"
        }
        
        emoji = emoji_map.get(level, "📢")
        
        message = {
            "msgtype": "markdown",
            "markdown": {
                "content": f"## {emoji} {title}\n\n{content}"
            }
        }
        
        # 发送请求
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"✓ 企业微信消息发送成功: {title}")
                else:
                    logger.error(f"✗ 企业微信消息发送失败: {result.get('errmsg')}")
            else:
                logger.error(f"✗ 企业微信消息发送失败: HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"✗ 企业微信消息发送异常: {str(e)}")
            raise


class DingTalkNotifier(BaseNotifier):
    """钉钉通知器"""
    
    def __init__(
        self,
        webhook_url: str,
        secret: Optional[str] = None,
        enabled: bool = True
    ):
        """
        初始化钉钉通知器
        
        Args:
            webhook_url: 钉钉机器人Webhook URL
            secret: 安全设置的密钥（如果启用了加签）
            enabled: 是否启用
        """
        super().__init__(enabled)
        self.webhook_url = webhook_url
        self.secret = secret
        logger.info("钉钉通知器初始化")
    
    def _do_send(self, title: str, content: str, level: NotificationLevel):
        """发送钉钉消息"""
        # 构建消息
        emoji_map = {
            NotificationLevel.INFO: "📘",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨"
        }
        
        emoji = emoji_map.get(level, "📢")
        
        message = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"{emoji} {title}",
                "text": f"### {emoji} {title}\n\n{content}"
            }
        }
        
        # 如果启用了加签，计算签名
        url = self.webhook_url
        if self.secret:
            import time
            import hmac
            import hashlib
            import base64
            import urllib.parse
            
            timestamp = str(round(time.time() * 1000))
            secret_enc = self.secret.encode('utf-8')
            string_to_sign = f'{timestamp}\n{self.secret}'
            string_to_sign_enc = string_to_sign.encode('utf-8')
            
            hmac_code = hmac.new(
                secret_enc,
                string_to_sign_enc,
                digestmod=hashlib.sha256
            ).digest()
            
            sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
            url = f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"
        
        # 发送请求
        try:
            response = requests.post(
                url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"✓ 钉钉消息发送成功: {title}")
                else:
                    logger.error(f"✗ 钉钉消息发送失败: {result.get('errmsg')}")
            else:
                logger.error(f"✗ 钉钉消息发送失败: HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"✗ 钉钉消息发送异常: {str(e)}")
            raise


class ConsoleNotifier(BaseNotifier):
    """控制台通知器（默认启用）"""
    
    def __init__(self, enabled: bool = True):
        """初始化控制台通知器"""
        super().__init__(enabled)
    
    def _do_send(self, title: str, content: str, level: NotificationLevel):
        """输出到控制台"""
        emoji_map = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🔴"
        }
        
        emoji = emoji_map.get(level, "📢")
        
        print(f"\n{emoji} [{level.value.upper()}] {title}")
        print("-" * 60)
        print(content)
        print("-" * 60 + "\n")


class NotificationManager:
    """通知管理器 - 统一管理多种通知方式"""
    
    def __init__(self):
        """初始化通知管理器"""
        self.notifiers: List[BaseNotifier] = []
        
        # 默认启用控制台通知
        self.add_notifier(ConsoleNotifier())
        
        logger.info("通知管理器初始化完成")
    
    def add_notifier(self, notifier: BaseNotifier):
        """
        添加通知器
        
        Args:
            notifier: 通知器实例
        """
        self.notifiers.append(notifier)
        logger.info(f"添加通知器: {notifier.__class__.__name__}")
    
    def send(
        self,
        title: str,
        content: str,
        level: NotificationLevel = NotificationLevel.INFO
    ):
        """
        发送通知到所有已启用的通知器
        
        Args:
            title: 标题
            content: 内容
            level: 通知级别
        """
        logger.info(f"发送通知: {title} [{level.value}]")
        
        for notifier in self.notifiers:
            if notifier.enabled:
                try:
                    notifier.send(title, content, level)
                except Exception as e:
                    logger.error(f"通知器 {notifier.__class__.__name__} 发送失败: {str(e)}")
    
    def send_signal_alert(self, stock_info: Dict, signal: Dict, period: str):
        """
        发送买卖点信号通知
        
        Args:
            stock_info: 股票信息
            signal: 信号信息
            period: 周期
        """
        signal_type = signal.get('type', 'unknown')
        price = signal.get('price', 0)
        time_str = signal.get('time', '').strftime('%Y-%m-%d %H:%M') if 'time' in signal else ''
        reason = signal.get('reason', '')
        
        # 判断是买入还是卖出
        action = "买入" if "buy" in signal_type else "卖出"
        
        title = f"【{action}信号】{stock_info['name']}({stock_info['code']})"
        
        content = f"""
**股票**: {stock_info['name']} ({stock_info['code']})
**周期**: {period}
**信号类型**: {signal_type}
**触发时间**: {time_str}
**触发价格**: ¥{price:.2f}
**原因**: {reason}

请及时关注！
        """.strip()
        
        level = NotificationLevel.WARNING if "买入" in action else NotificationLevel.INFO
        
        self.send(title, content, level)


# 全局通知管理器实例
_notification_manager = None


def get_notification_manager() -> NotificationManager:
    """获取全局通知管理器实例"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


# 便捷函数
def send_notification(
    title: str,
    content: str,
    level: NotificationLevel = NotificationLevel.INFO
):
    """
    快速发送通知
    
    Args:
        title: 标题
        content: 内容
        level: 级别
    """
    manager = get_notification_manager()
    manager.send(title, content, level)
