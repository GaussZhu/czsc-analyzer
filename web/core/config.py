"""
Web应用配置模块
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class WebConfig:
    """Web应用配置"""
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # 路径配置
    base_dir: Path = Path(__file__).parent.parent
    static_dir: Path = base_dir / "static"
    templates_dir: Path = base_dir / "templates"
    
    # 应用配置
    title: str = "CZSC股票分析工具"
    version: str = "2.1.0"
    description: str = "基于缠论的股票技术分析平台"
    
    # 域名配置
    domain: str = "www.chanquant.com"
    
    # 功能开关
    enable_tracking: bool = True
    enable_recommend: bool = True
    enable_forecast: bool = True
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("WEB_HOST", "0.0.0.0"),
            port=int(os.getenv("WEB_PORT", "8000")),
            reload=os.getenv("WEB_RELOAD", "false").lower() == "true",
        )

# 全局配置实例
_config: Optional[WebConfig] = None

def get_web_config() -> WebConfig:
    """获取Web配置单例"""
    global _config
    if _config is None:
        _config = WebConfig.from_env()
    return _config
