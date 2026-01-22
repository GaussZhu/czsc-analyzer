#!/usr/bin/env python3
"""
CZSC Web 应用服务启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import uvicorn
from web.main import app

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                 CZSC 股票分析工具 v2.1                             ║
║                 启动 Web 服务...                                   ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
