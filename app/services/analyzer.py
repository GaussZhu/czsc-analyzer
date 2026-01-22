# -*- coding: utf-8 -*-
"""
Analysis Service Module
分析服务 - 整合所有CZSC分析功能
"""

import pandas as pd
import datetime
import os
from glob import glob
from typing import Dict, Optional

from app.core.data_source import UnifiedDataSource, DataSourceType
from app.core.indicators import TechnicalIndicators
from app.core.fractal import FractalAnalyzer
from app.core.stroke import StrokeAnalyzer
from app.core.pivot import PivotAnalyzer
from app.core.divergence import DivergenceAnalyzer
from app.core.signals import SignalAnalyzer
from app.config.settings import get_config
from app.config.logging_config import get_analysis_logger

logger = get_analysis_logger()

# ========== 基本面雷达数据加载 ==========
_BUFFETT_RADAR_MAP = None

def load_buffett_radar_info():
    """加载基本面评分数据"""
    base_dir = "/root/stock_monitor_project/webapp/web/static/"
    files = sorted(glob(f"{base_dir}/buffett_analysis_full_*.csv"), reverse=True)
    if not files:
        logger.warning("未找到 buffett_analysis_full_*.csv 文件")
        return {}
    
    latest_file = files[0]
    logger.info(f"加载基本面数据: {latest_file}")
    df = pd.read_csv(latest_file)
    
    # 确保股票代码是字符串，并移除前缀
    df["stock_code"] = df["stock_code"].astype(str)
    df["stock_code"] = df["stock_code"].str.replace("sh.", "", regex=False)
    df["stock_code"] = df["stock_code"].str.replace("sz.", "", regex=False)
    
    # 去重：保留每个 stock_code 第一条记录
    df = df.drop_duplicates(subset="stock_code", keep="first")
    df = df.set_index("stock_code")
    return df.to_dict("index")

def get_radar_data(code: str):
    """
    获取股票的基本面雷达数据
    
    Args:
        code: 股票代码
        
    Returns:
        (labels, values) 雷达图的标签和数据
    """
    global _BUFFETT_RADAR_MAP
    if _BUFFETT_RADAR_MAP is None:
        _BUFFETT_RADAR_MAP = load_buffett_radar_info()
    
    # 前缀兼容防御性处理
    code = str(code).replace("sh.", "").replace("sz.", "")
    
    data = _BUFFETT_RADAR_MAP.get(code)
    if not data:
        logger.debug(f"[雷达图] 未找到评分数据: {code}")
        return [], []
    
    logger.info(f"[雷达图] 成功获取评分: {code}")
    
    labels = ["稳健型", "成长型", "现金流", "估值吸引", "分红型", "效率型"]
    values = [
        round((data.get("score_profit_stability", 0) + data.get("score_roe", 0) + data.get("score_debt_ratio", 0)) / 3, 2),
        round(data.get("score_profit_growth", 0), 2),
        round(data.get("score_fcf", 0), 2),
        round(data.get("score_valuation", 0), 2),
        round(data.get("score_dividend", 0), 2),
        round(data.get("score_inventory_turnover", 0), 2)
    ]
    return labels, values


class StockAnalyzer:
    """股票分析服务"""
    
    def __init__(self, use_mysql: bool = None):
        """
        初始化股票分析服务
        
        Args:
            use_mysql: 是否使用MySQL数据源，None则根据配置自动决定
        """
        self.config = get_config()
        
        # 决定数据源类型
        if use_mysql is None:
            source_type = DataSourceType.AUTO
        elif use_mysql:
            source_type = DataSourceType.MYSQL
        else:
            source_type = DataSourceType.TDX
        
        # 初始化统一数据源（支持TDX和MySQL）
        self.data_source = UnifiedDataSource(source_type)
        
        # 初始化各个分析组件
        self.indicators = TechnicalIndicators()
        self.fractal_analyzer = FractalAnalyzer()
        self.stroke_analyzer = StrokeAnalyzer()
        self.pivot_analyzer = PivotAnalyzer()
        self.divergence_analyzer = DivergenceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        
        logger.info(f"股票分析服务初始化完成，数据源: {source_type.value}")
    
    def analyze_stock_with_date_filter(
        self,
        stock_code: str,
        stock_name: str,
        market: int,
        end_date: datetime = None,
        period: str = 'daily'
    ) -> Optional[Dict]:
        """
        分析股票（支持指定截止日期，用于回测）
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            market: 市场代码
            end_date: 数据截止日期（用于回测）
            period: 数据周期 ('daily' 或 '30min')
        
        Returns:
            Dict: 分析结果，包含买卖点、中枢、背驰等信息
        """
        logger.info(f"开始分析股票: {stock_name}({stock_code}), 周期: {period}, 截止: {end_date}")
        
        try:
            # 1. 获取数据
            data = self.data_source.get_stock_data(
                stock_code, 
                market, 
                period,
                prefer_mysql=True
            )
            
            if data is None or len(data) < 30:
                logger.warning(f"数据不足，无法分析: {stock_code}")
                return None
            
            # 2. 如果指定了截止日期，过滤数据
            if end_date is not None:
                # 确保date列存在并转换为datetime
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[data['date'] <= pd.Timestamp(end_date)]
                    # 重置索引，避免后续操作出问题
                    data = data.reset_index(drop=True)
                else:
                    # 如果date不是列，可能是索引
                    data.index = pd.to_datetime(data.index)
                    data = data[data.index <= pd.Timestamp(end_date)]
                
                # 只保留最近300条
                if len(data) > 300:
                    data = data.iloc[-300:]
                
                logger.info(f"过滤后数据: {len(data)} 条（截止 {end_date}）")
            
            if len(data) < 30:
                logger.warning(f"过滤后数据不足: {stock_code}")
                return None
            
            logger.debug(f"获取到 {len(data)} 条数据")
            
            # 3. 计算技术指标
            data = self.indicators.calculate_macd(data)
            
            # 4. 识别分型
            data = self.fractal_analyzer.identify_fractals(data)
            
            # 5. 识别笔
            strokes = self.stroke_analyzer.identify_strokes(data)
            
            # 6. 识别中枢
            pivot_points = self.pivot_analyzer.identify_pivots(strokes)
            
            # 7. 检测背驰
            divergences = self.divergence_analyzer.detect_divergences(data, strokes)
            
            # 8. 识别买卖点
            signals = self.signal_analyzer.identify_signals(data, pivot_points, divergences)
            
            # 9. 获取最后价格（列名可能是 'close' 或 'Close'）
            close_col = 'close' if 'close' in data.columns else ('Close' if 'Close' in data.columns else None)
            if close_col:
                last_price = float(data.iloc[-1][close_col]) if len(data) > 0 else 0
            else:
                last_price = 0
                logger.warning(f"数据中没有close列: {data.columns.tolist()}")
            
            # 10. 返回结果
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'last_price': last_price,
                'data_count': len(data),
                'signals': signals,
                'fractals': {
                    'top': len(data[data.get('FractalTop', False) if 'FractalTop' in data.columns else data.get('fractal_top', False)]) if 'FractalTop' in data.columns or 'fractal_top' in data.columns else 0,
                    'bottom': len(data[data.get('FractalBottom', False) if 'FractalBottom' in data.columns else data.get('fractal_bottom', False)]) if 'FractalBottom' in data.columns or 'fractal_bottom' in data.columns else 0
                },
                'strokes': strokes,
                'pivot_points': pivot_points,
                'divergences': divergences
            }
            
        except Exception as e:
            logger.error(f"分析股票失败: {stock_code}, 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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
            # 1. 获取数据（优先使用MySQL，失败则使用TDX）
            data = self.data_source.get_stock_data(
                stock_code, 
                market, 
                period,
                prefer_mysql=True
            )
            
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
            
            # 8. 不过滤信号，显示所有历史买卖点
            # 注释掉过滤逻辑，以便在分析页面显示完整的历史数据
            # if filter_recent:
            #     signals = self._filter_recent_signals(signals, period)
            
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
    
    def get_30min_tracking(self, stock_code: str, market: int, buy_signals: list) -> list:
        """
        获取买点的30分钟追踪状态
        
        Args:
            stock_code: 股票代码
            market: 市场代码
            buy_signals: 买点列表
        
        Returns:
            list: 追踪状态列表
        """
        tracking_list = []
        
        try:
            # 获取30分钟K线数据
            data_30min = self.data_source.get_stock_data(
                stock_code, market, '30min', prefer_mysql=True
            )
            
            if data_30min is None or len(data_30min) == 0:
                logger.warning(f"无法获取30分钟数据: {stock_code}")
                return []
            
            for signal in buy_signals:
                signal_date = pd.to_datetime(signal['time'])
                
                # 筛选信号日期之后的30分钟K线
                tracking_data = data_30min[data_30min.index >= signal_date]
                
                if len(tracking_data) > 0:
                    current_price = float(tracking_data['Close'].iloc[-1])
                    buy_price = float(signal['price'])
                    
                    # 计算涨幅
                    change_pct = (current_price - buy_price) / buy_price * 100
                    
                    # 计算最大涨幅
                    max_price = float(tracking_data['High'].max())
                    max_change = (max_price - buy_price) / buy_price * 100
                    
                    # 计算最大回撤
                    peak_prices = tracking_data['High'].cummax()
                    drawdowns = (tracking_data['Close'] - peak_prices) / peak_prices * 100
                    max_drawdown = float(drawdowns.min())
                    
                    # 生成建议
                    if change_pct > 5 and max_drawdown < -3:
                        suggestion = "考虑止盈"
                    elif change_pct < -3:
                        suggestion = "考虑止损"
                    elif change_pct > 3:
                        suggestion = "持续盈利，继续持有"
                    else:
                        suggestion = "继续持有"
                    
                    tracking_list.append({
                        "base_signal": {
                            "类型": signal['type'],
                            "时间": str(signal_date.date())
                        },
                        "tracking_status": {
                            "当前价格": round(current_price, 2),
                            "涨幅": f"{change_pct:+.2f}%",
                            "最大涨幅": f"{max_change:+.2f}%",
                            "最大回撤": f"{max_drawdown:.2f}%",
                            "最新30分钟K线时间": str(tracking_data.index[-1]),
                            "当前建议": suggestion
                        }
                    })
            
            return tracking_list
            
        except Exception as e:
            logger.error(f"30分钟追踪失败 {stock_code}: {str(e)}", exc_info=True)
            return []
    
    def analyze_trend(self, pivot_points: list, data: pd.DataFrame) -> dict:
        """
        分析当前趋势
        
        Args:
            pivot_points: 中枢列表
            data: K线数据
        
        Returns:
            dict: 趋势分析结果
        """
        try:
            if len(data) == 0:
                return None
            
            current_price = float(data['Close'].iloc[-1])
            
            if len(pivot_points) < 1:
                return {
                    "trend_type": "趋势形成中",
                    "position": "数据不足",
                    "detail": "中枢数量不足，无法判断趋势",
                    "signal": "等待更多数据"
                }
            
            # 获取最近的中枢
            latest_pivot = pivot_points[-1]
            pivot_high = latest_pivot.get('high', 0)
            pivot_low = latest_pivot.get('low', 0)
            pivot_mid = (pivot_high + pivot_low) / 2
            
            # 判断价格相对中枢的位置
            if current_price > pivot_high:
                position = "向上离开中枢"
                price_distance = ((current_price - pivot_high) / pivot_high * 100)
                detail = f"当前价格高于中枢上沿{price_distance:.2f}%"
            elif current_price < pivot_low:
                position = "向下离开中枢"
                price_distance = ((pivot_low - current_price) / pivot_low * 100)
                detail = f"当前价格低于中枢下沿{price_distance:.2f}%"
            elif current_price > pivot_mid:
                position = "中枢上沿"
                detail = "价格在中枢上半部分"
            else:
                position = "中枢下沿"
                detail = "价格在中枢下半部分"
            
            # 判断整体趋势
            if len(pivot_points) >= 3:
                recent_pivots = pivot_points[-3:]
                pivot_highs = [p.get('high', 0) for p in recent_pivots]
                pivot_lows = [p.get('low', 0) for p in recent_pivots]
                
                # 判断中枢高低点是否递增/递减
                highs_trend = all(pivot_highs[i] > pivot_highs[i-1] for i in range(1, len(pivot_highs)))
                lows_trend = all(pivot_lows[i] > pivot_lows[i-1] for i in range(1, len(pivot_lows)))
                
                if highs_trend and lows_trend:
                    trend_type = "上升趋势"
                    signal = "多头趋势，关注买点"
                elif all(pivot_highs[i] < pivot_highs[i-1] for i in range(1, len(pivot_highs))) and \
                     all(pivot_lows[i] < pivot_lows[i-1] for i in range(1, len(pivot_lows))):
                    trend_type = "下降趋势"
                    signal = "空头趋势，关注卖点"
                else:
                    trend_type = "盘整"
                    signal = "震荡走势，等待突破"
            else:
                trend_type = "趋势形成中"
                signal = "等待更多中枢形成"
            
            return {
                "trend_type": trend_type,
                "position": position,
                "detail": detail,
                "signal": signal
            }
            
        except Exception as e:
            logger.error(f"趋势分析失败: {str(e)}", exc_info=True)
            return None
    
    def __del__(self):
        """析构函数：确保连接关闭"""
        if hasattr(self, 'data_source'):
            self.data_source.disconnect()


# 便捷函数
def analyze_stock(stock_code: str, stock_name: str, market: int, period: str = 'daily') -> Optional[Dict]:
    """快速分析股票"""
    analyzer = StockAnalyzer()
    return analyzer.analyze_stock(stock_code, stock_name, market, period)
