# -*- coding: utf-8 -*-
"""
CZSC Stock Analyzer - Main Application Entry Point
缠中说禅股票分析工具 - 主程序入口
"""

import sys
import argparse
from pytdx.params import TDXParams

from app.services.monitor import StockMonitor
from app.services.analyzer import StockAnalyzer
from app.config.settings import get_config, update_stock_list
from app.config.logging_config import setup_logging, get_logger

# 初始化日志
setup_logging()
logger = get_logger(__name__)


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              CZSC Stock Analyzer (缠中说禅股票分析工具)                    ║
║                         Version 2.0 - Refactored                          ║
║                                                                           ║
║  基于缠中说禅理论的智能股票分析系统                                         ║
║  - 分型识别  - 笔的识别  - 中枢识别                                        ║
║  - 背驰检测  - 买卖点识别  - 自动监控                                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def cmd_monitor(args):
    """监控模式"""
    config = get_config()
    
    print_banner()
    print(f"\n启动监控模式...")
    print(f"监控间隔: {args.interval}秒")
    print(f"监控股票数量: {len(config.stock_list.stocks)}\n")
    
    for stock in config.stock_list.stocks:
        print(f"  - {stock['name']} ({stock['code']})")
    
    print("\n按 Ctrl+C 停止监控\n")
    
    monitor = StockMonitor()
    monitor.monitor_stocks(interval=args.interval)


def cmd_scan(args):
    """扫描模式"""
    config = get_config()
    
    print_banner()
    print(f"\n执行一次性扫描...")
    
    monitor = StockMonitor()
    results = monitor.scan_once()
    
    print(f"\n{'='*80}")
    print(f"扫描完成！共分析 {len(results)} 只股票")
    print(f"{'='*80}\n")


def cmd_analyze(args):
    """分析单只股票"""
    print_banner()
    print(f"\n分析股票: {args.name} ({args.code})")
    
    analyzer = StockAnalyzer()
    
    # 日线分析
    if args.daily or args.all:
        print("\n[日线分析]")
        daily_result = analyzer.analyze_stock(
            args.code,
            args.name,
            args.market,
            'daily'
        )
        if daily_result:
            analyzer.print_analysis_results(daily_result)
    
    # 30分钟线分析
    if args.min30 or args.all:
        print("\n[30分钟线分析]")
        min30_result = analyzer.analyze_stock(
            args.code,
            args.name,
            args.market,
            '30min'
        )
        if min30_result:
            analyzer.print_analysis_results(min30_result)


def cmd_config(args):
    """配置管理"""
    config = get_config()
    
    print_banner()
    
    if args.show:
        print("\n当前配置：")
        print(f"\n监控配置:")
        print(f"  - 监控间隔: {config.monitor.interval}秒")
        print(f"  - 日线信号范围: 最近{config.monitor.recent_days}天")
        print(f"  - 分钟线信号范围: 最近{config.monitor.recent_hours}小时")
        
        print(f"\nMACD配置:")
        print(f"  - 快速EMA: {config.macd.fast}")
        print(f"  - 慢速EMA: {config.macd.slow}")
        print(f"  - 信号线: {config.macd.signal}")
        
        print(f"\n监控股票列表:")
        for stock in config.stock_list.stocks:
            print(f"  - {stock['name']} ({stock['code']})")
    
    if args.list_stocks:
        print("\n监控股票列表:")
        for i, stock in enumerate(config.stock_list.stocks, 1):
            market_name = "上海" if stock['market'] == TDXParams.MARKET_SH else "深圳"
            print(f"  {i}. {stock['name']} ({stock['code']}) - {market_name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CZSC股票分析工具 - 基于缠中说禅理论',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动监控模式（默认5分钟刷新）
  python -m app.main monitor
  
  # 自定义监控间隔（10分钟）
  python -m app.main monitor --interval 600
  
  # 执行一次性扫描
  python -m app.main scan
  
  # 分析单只股票（日线）
  python -m app.main analyze --code 600000 --name 浦发银行 --market 1 --daily
  
  # 分析单只股票（全部周期）
  python -m app.main analyze --code 600000 --name 浦发银行 --market 1 --all
  
  # 查看配置
  python -m app.main config --show
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 监控命令
    parser_monitor = subparsers.add_parser('monitor', help='启动监控模式')
    parser_monitor.add_argument(
        '--interval',
        type=int,
        default=300,
        help='监控间隔（秒），默认300秒'
    )
    parser_monitor.set_defaults(func=cmd_monitor)
    
    # 扫描命令
    parser_scan = subparsers.add_parser('scan', help='执行一次性扫描')
    parser_scan.set_defaults(func=cmd_scan)
    
    # 分析命令
    parser_analyze = subparsers.add_parser('analyze', help='分析单只股票')
    parser_analyze.add_argument('--code', required=True, help='股票代码')
    parser_analyze.add_argument('--name', required=True, help='股票名称')
    parser_analyze.add_argument(
        '--market',
        type=int,
        default=TDXParams.MARKET_SH,
        help='市场代码（0=深圳, 1=上海）'
    )
    parser_analyze.add_argument('--daily', action='store_true', help='分析日线')
    parser_analyze.add_argument('--min30', action='store_true', help='分析30分钟线')
    parser_analyze.add_argument('--all', action='store_true', help='分析所有周期')
    parser_analyze.set_defaults(func=cmd_analyze)
    
    # 配置命令
    parser_config = subparsers.add_parser('config', help='配置管理')
    parser_config.add_argument('--show', action='store_true', help='显示当前配置')
    parser_config.add_argument('--list-stocks', action='store_true', help='列出监控股票')
    parser_config.set_defaults(func=cmd_config)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行错误: {str(e)}", exc_info=True)
        print(f"\n✗ 错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
