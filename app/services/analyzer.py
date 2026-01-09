# -*- coding: utf-8 -*-
"""
Analysis Service Module
分析服务 - 整合所有CZSC分析功能
"""

import pandas as pd
import datetime
from typing import Dict, Optional

from app.core.data_fetcher import TDXDataFetcher
from app.core.indicators import TechnicalIndicators
from app.core.fractal import FractalAnalyzer
from app.core.stroke import StrokeAnalyzer
from app.core.pivot import PivotAnalyzer
from app.core.divergence import DivergenceAnalyzer
from app.core.signals import SignalAnalyzer
from app.config.settings import get_config
from app.config.logging_config import get_analysis_logger

logger = get_analysis_logger()


class StockAnalyzer:
    """股票分析服务"""
    
    def __init__(self):
        """初始化股票分析服务"""
        self.config = get_config()
        
        # 初始化各个组件
        self.data_fetcher = TDXDataFetcher()
        self.indicators = TechnicalIndicators()
        self.fractal_analyzer = FractalAnalyzer()
        self.stroke_analyzer = StrokeAnalyzer()
        self.pivot_analyzer = PivotAnalyzer()
        self.divergence_analyzer = DivergenceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        
        logger.info("股票分析服务初始化完成")
    
    def analyze_stock(
        self,
        stock_code: str,
        stock_name: str,
        market: int,
        period: str = 'daily'
    ) -> Optional[Dict]:
        """
        完整分析一只股票
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            market: 市场代码
            period: 数据周期 ('daily' 或 '30min')
        
        Returns:
            Dict: 分析结果，包含买卖点、中枢、背驰等信息
        """
        logger.info(f"开始分析股票: {stock_name}({stock_code}), 周期: {period}")
        
        try:
            # 1. 获取数据
            data = self.data_fetcher.get_stock_data(stock_code, market, period)
            
            if data is None or len(data) < 30:
                logger.warning(f"数据不足，无法分析: {stock_code}")
                return None
            
            logger.debug(f"获取到 {len(data)} 条数据")
            
            # 2. 计算技术指标
            data = self.indicators.calculate_macd(data)
            
            # 3. 识别分型
            data = self.fractal_analyzer.identify_fractals(data)
            
            # 4. 识别笔
            strokes = self.stroke_analyzer.identify_strokes(data)
            
            # 5. 识别中枢
            pivot_points = self.pivot_analyzer.identify_pivots(strokes)
            
            # 6. 检测背驰
            divergences = self.divergence_analyzer.detect_divergences(data, strokes)
            
            # 7. 识别买卖点
            signals = self.signal_analyzer.identify_signals(data, pivot_points, divergences)
            
            # 8. 过滤最近的信号
            signals = self._filter_recent_signals(signals, period)
            
            # 9. 准备结果
            result = {
                'stock': {
                    'code': stock_code,
                    'name': stock_name,
                    'market': market
                },
                'period': period,
                'last_price': float(data['Close'].iloc[-1]),
                'last_update': data.index[-1],
                'data_count': len(data),
                'strokes_count': len(strokes),
                'pivot_points': pivot_points,
                'divergences': divergences,
                'signals': signals,
                'analysis_time': datetime.datetime.now()
            }
            
            logger.info(
                f"分析完成: {stock_name}({stock_code}), "
                f"买点={sum(len(v) for v in signals['buy'].values())}, "
                f"卖点={sum(len(v) for v in signals['sell'].values())}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"分析股票失败 {stock_code}: {str(e)}", exc_info=True)
            return None
    
    def _filter_recent_signals(self, signals: Dict, period: str) -> Dict:
        """
        过滤最近的信号
        
        Args:
            signals: 原始信号
            period: 周期类型
        
        Returns:
            Dict: 过滤后的信号
        """
        if period == 'daily':
            # 日线：最近3天
            time_delta = datetime.timedelta(days=self.config.monitor.recent_days)
        else:
            # 分钟线：最近6小时
            time_delta = datetime.timedelta(hours=self.config.monitor.recent_hours)
        
        valid_start = pd.Timestamp.now() - time_delta
        
        filtered_signals = {
            'buy': {},
            'sell': {},
            'valid_start': valid_start,
            'valid_end': pd.Timestamp.now()
        }
        
        # 过滤买点
        for signal_type, points in signals['buy'].items():
            filtered_signals['buy'][signal_type] = [
                p for p in points if p['time'] > valid_start
            ]
        
        # 过滤卖点
        for signal_type, points in signals['sell'].items():
            filtered_signals['sell'][signal_type] = [
                p for p in points if p['time'] > valid_start
            ]
        
        return filtered_signals
    
    def print_analysis_results(self, results: Dict, max_display: int = 3):
        """
        打印分析结果（优化显示）
        
        Args:
            results: 分析结果
            max_display: 最多显示的信号数量
        """
        if results is None:
            logger.warning("无分析结果可显示")
            return
        
        print(f"\n{'='*80}")
        print(f"股票: {results['stock']['name']} ({results['stock']['code']})")
        print(f"周期: {results['period']} | 最新价: {results['last_price']:.2f}")
        print(f"数据时间: {results['last_update']}")
        print('-'*80)
        
        # 关键信号摘要
        buy_count = sum(len(v) for v in results['signals']['buy'].values())
        sell_count = sum(len(v) for v in results['signals']['sell'].values())
        
        print(f"\n关键信号摘要：")
        print(f"★ 发现 {buy_count} 个买入信号 | {sell_count} 个卖出信号")
        print(f"★ 中枢数量: {len(results['pivot_points'])}")
        print(f"★ 背驰数量: {len(results['divergences'])}")
        
        # 合并所有信号并按时间排序
        all_signals = []
        
        for stype, points in results['signals']['buy'].items():
            for p in points:
                all_signals.append({
                    'time': p['time'],
                    'price': p['price'],
                    'type': f'买点_{stype}',
                    'reason': p.get('reason', '')
                })
        
        for stype, points in results['signals']['sell'].items():
            for p in points:
                all_signals.append({
                    'time': p['time'],
                    'price': p['price'],
                    'type': f'卖点_{stype}',
                    'reason': p.get('reason', '')
                })
        
        # 按时间倒序排列
        all_signals.sort(key=lambda x: x['time'], reverse=True)
        
        # 显示最近信号
        if all_signals:
            print(f"\n最新信号（最近{max_display}个）：")
            for signal in all_signals[:max_display]:
                time_str = signal['time'].strftime('%Y-%m-%d %H:%M')
                print(f"→ {time_str} | {signal['type']} | 价格: {signal['price']:.2f}")
                print(f"   原因: {signal['reason']}")
        else:
            print("\n✓ 当前时间段内无信号")
        
        # 显示重要背驰
        if results['divergences']:
            print(f"\n重要背驰：")
            for d in results['divergences'][:3]:
                div_type = '看涨' if d['type'] == 'bullish' else '看跌'
                time_str = d['time'].strftime('%Y-%m-%d %H:%M')
                print(f"❗ {time_str} | {div_type}背驰 | 价格: {d['price']:.2f}")
        
        print(f"\n{'='*80}\n")
    
    def batch_analyze(self, stock_list: list, period: str = 'daily') -> list:
        """
        批量分析多只股票
        
        Args:
            stock_list: 股票列表
            period: 数据周期
        
        Returns:
            list: 分析结果列表
        """
        results = []
        
        logger.info(f"开始批量分析 {len(stock_list)} 只股票")
        
        for stock in stock_list:
            result = self.analyze_stock(
                stock['code'],
                stock['name'],
                stock['market'],
                period
            )
            
            if result:
                results.append(result)
        
        logger.info(f"批量分析完成: 成功 {len(results)}/{len(stock_list)}")
        
        return results
    
    def get_signals_summary(self, results: Dict) -> Dict:
        """
        获取信号摘要
        
        Args:
            results: 分析结果
        
        Returns:
            Dict: 信号摘要
        """
        if not results:
            return {}
        
        buy_count = sum(len(v) for v in results['signals']['buy'].values())
        sell_count = sum(len(v) for v in results['signals']['sell'].values())
        
        has_buy = buy_count > 0
        has_sell = sell_count > 0
        
        return {
            'stock': results['stock'],
            'last_price': results['last_price'],
            'buy_count': buy_count,
            'sell_count': sell_count,
            'has_buy_signal': has_buy,
            'has_sell_signal': has_sell,
            'pivot_count': len(results['pivot_points']),
            'divergence_count': len(results['divergences']),
            'analysis_time': results['analysis_time']
        }
    
    def __del__(self):
        """析构函数：确保连接关闭"""
        if hasattr(self, 'data_fetcher'):
            self.data_fetcher.disconnect()


# 便捷函数
def analyze_stock(stock_code: str, stock_name: str, market: int, period: str = 'daily') -> Optional[Dict]:
    """快速分析股票"""
    analyzer = StockAnalyzer()
    return analyzer.analyze_stock(stock_code, stock_name, market, period)
