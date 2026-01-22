#!/usr/bin/env python3
"""
Web服务器启动脚本
用于启动CZSC Web应用
"""
import uvicorn
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='CZSC Web服务器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--reload', action='store_true', help='启用热重载（开发模式）')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║                     CZSC 股票分析工具 Web 服务器                               ║
║                     CZSC Stock Analyzer Web Server                             ║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 启动信息:
   - 地址: http://{args.host}:{args.port}
   - 模式: {'开发模式 (热重载)' if args.reload else '生产模式'}
   - 工作进程: {args.workers}

📝 API文档:
   - Swagger UI: http://{args.host}:{args.port}/docs
   - ReDoc: http://{args.host}:{args.port}/redoc

🔧 配置文件: .env

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

按 Ctrl+C 停止服务器

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    uvicorn.run(
        "web.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )

if __name__ == "__main__":
    main()
