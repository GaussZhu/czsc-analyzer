# -*- coding: utf-8 -*-
"""
Trading Signals Module
买卖点识别模块 - CZSC理论核心组件
"""

import pandas as pd
from typing import List, Tuple, Dict
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class SignalAnalyzer:
    """买卖点分析器"""
    
    def __init__(self):
        """初始化买卖点分析器"""
        logger.debug("买卖点分析器初始化完成")
    
    def identify_signals(
        self,
        data: pd.DataFrame,
        pivot_points: List[Dict],
        divergences: List[Dict]
    ) -> Dict:
        """
        识别买卖点
        
        Args:
            data: 价格数据
            pivot_points: 中枢列表
            divergences: 背驰列表
        
        Returns:
            Dict: {'buy': {...}, 'sell': {...}}
        """
        buy_points = {
            'first_type': [],   # 第一类买点
            'second_type': [],  # 第二类买点
            'third_type': []    # 第三类买点
        }
        
        sell_points = {
            'first_type': [],   # 第一类卖点
            'second_type': [],  # 第二类卖点
            'third_type': []    # 第三类卖点
        }
        
        try:
            # 识别第一类买卖点（基于背驰）
            buy_points['first_type'] = self._identify_first_type_buy(divergences)
            sell_points['first_type'] = self._identify_first_type_sell(divergences)
            
            # 识别第二类买卖点（基于第一类点的回调）
            buy_points['second_type'] = self._identify_second_type_buy(
                data, buy_points['first_type']
            )
            sell_points['second_type'] = self._identify_second_type_sell(
                data, sell_points['first_type']
            )
            
            # 识别第三类买卖点（基于中枢突破）
            buy_points['third_type'] = self._identify_third_type_buy(
                data, pivot_points
            )
            sell_points['third_type'] = self._identify_third_type_sell(
                data, pivot_points
            )
            
            total_buy = sum(len(v) for v in buy_points.values())
            total_sell = sum(len(v) for v in sell_points.values())
            
            logger.info(f"买卖点识别完成: 买点={total_buy}, 卖点={total_sell}")
            
            return {'buy': buy_points, 'sell': sell_points}
        
        except Exception as e:
            logger.error(f"买卖点识别失败: {str(e)}", exc_info=True)
            return {'buy': buy_points, 'sell': sell_points}
    
    def _identify_first_type_buy(self, divergences: List[Dict]) -> List[Dict]:
        """
        识别第一类买点（趋势反转点，伴随看涨背驰）
        
        Args:
            divergences: 背驰列表
        
        Returns:
            List[Dict]: 第一类买点列表
        """
        buy_points = []
        
        for div in divergences:
            if div['type'] == 'bullish':
                buy_point = {
                    'time': div['time'],
                    'price': div['price'],
                    'reason': '下跌趋势结束，出现看涨背驰',
                    'strength': div.get('strength', 'medium'),
                    'macd': div.get('macd')
                }
                buy_points.append(buy_point)
                
                logger.debug(f"第一类买点: {div['time']}, 价格: {div['price']:.2f}")
        
        return buy_points
    
    def _identify_first_type_sell(self, divergences: List[Dict]) -> List[Dict]:
        """
        识别第一类卖点（趋势反转点，伴随看跌背驰）
        
        Args:
            divergences: 背驰列表
        
        Returns:
            List[Dict]: 第一类卖点列表
        """
        sell_points = []
        
        for div in divergences:
            if div['type'] == 'bearish':
                sell_point = {
                    'time': div['time'],
                    'price': div['price'],
                    'reason': '上涨趋势结束，出现看跌背驰',
                    'strength': div.get('strength', 'medium'),
                    'macd': div.get('macd')
                }
                sell_points.append(sell_point)
                
                logger.debug(f"第一类卖点: {div['time']}, 价格: {div['price']:.2f}")
        
        return sell_points
    
    def _identify_second_type_buy(
        self,
        data: pd.DataFrame,
        first_type_buys: List[Dict]
    ) -> List[Dict]:
        """
        识别第二类买点（第一类买点后的回调低点）
        
        Args:
            data: 价格数据
            first_type_buys: 第一类买点列表
        
        Returns:
            List[Dict]: 第二类买点列表
        """
        buy_points = []
        
        for first_buy in first_type_buys:
            try:
                # 获取第一类买点的位置
                idx = data.index.get_loc(first_buy['time'])
                
                # 需要至少5根K线后才能观察回调
                if idx + 5 >= len(data):
                    continue
                
                # 在后续5-20根K线中寻找回调低点
                search_end = min(idx + 20, len(data))
                min_price = float('inf')
                min_idx = None
                
                for j in range(idx + 5, search_end):
                    if data['Low'].iloc[j] < min_price:
                        min_price = data['Low'].iloc[j]
                        min_idx = j
                
                # 回调低点必须高于第一类买点
                if min_idx and min_price > first_buy['price']:
                    buy_point = {
                        'time': data.index[min_idx],
                        'price': min_price,
                        'reason': f"第一类买点后的回调（第一类买点: {first_buy['time']}）",
                        'related_first_type': first_buy['time']
                    }
                    buy_points.append(buy_point)
                    
                    logger.debug(f"第二类买点: {data.index[min_idx]}, 价格: {min_price:.2f}")
            
            except (KeyError, IndexError):
                continue
        
        return buy_points
    
    def _identify_second_type_sell(
        self,
        data: pd.DataFrame,
        first_type_sells: List[Dict]
    ) -> List[Dict]:
        """
        识别第二类卖点（第一类卖点后的反弹高点）
        
        Args:
            data: 价格数据
            first_type_sells: 第一类卖点列表
        
        Returns:
            List[Dict]: 第二类卖点列表
        """
        sell_points = []
        
        for first_sell in first_type_sells:
            try:
                idx = data.index.get_loc(first_sell['time'])
                
                if idx + 5 >= len(data):
                    continue
                
                search_end = min(idx + 20, len(data))
                max_price = float('-inf')
                max_idx = None
                
                for j in range(idx + 5, search_end):
                    if data['High'].iloc[j] > max_price:
                        max_price = data['High'].iloc[j]
                        max_idx = j
                
                # 反弹高点必须低于第一类卖点
                if max_idx and max_price < first_sell['price']:
                    sell_point = {
                        'time': data.index[max_idx],
                        'price': max_price,
                        'reason': f"第一类卖点后的反弹（第一类卖点: {first_sell['time']}）",
                        'related_first_type': first_sell['time']
                    }
                    sell_points.append(sell_point)
                    
                    logger.debug(f"第二类卖点: {data.index[max_idx]}, 价格: {max_price:.2f}")
            
            except (KeyError, IndexError):
                continue
        
        return sell_points
    
    def _identify_third_type_buy(
        self,
        data: pd.DataFrame,
        pivot_points: List[Dict]
    ) -> List[Dict]:
        """
        识别第三类买点（中枢下沿附近的支撑点）
        
        Args:
            data: 价格数据
            pivot_points: 中枢列表
        
        Returns:
            List[Dict]: 第三类买点列表
        """
        buy_points = []
        
        for pivot in pivot_points:
            try:
                idx = data.index.get_loc(pivot['end'])
                
                # 在中枢结束后的5-30根K线中寻找
                search_end = min(idx + 30, len(data))
                
                for j in range(idx + 5, search_end):
                    low_price = data['Low'].iloc[j]
                    
                    # 价格接近但未跌破中枢下沿（允许5%的误差）
                    if pivot['low'] * 0.95 < low_price <= pivot['low'] * 1.05:
                        buy_point = {
                            'time': data.index[j],
                            'price': low_price,
                            'reason': f"测试中枢下沿支撑（中枢: {pivot['low']:.2f}）",
                            'pivot_low': pivot['low']
                        }
                        buy_points.append(buy_point)
                        
                        logger.debug(f"第三类买点: {data.index[j]}, 价格: {low_price:.2f}")
                        break  # 每个中枢只找一个第三类买点
            
            except (KeyError, IndexError):
                continue
        
        return buy_points
    
    def _identify_third_type_sell(
        self,
        data: pd.DataFrame,
        pivot_points: List[Dict]
    ) -> List[Dict]:
        """
        识别第三类卖点（中枢上沿附近的阻力点）
        
        Args:
            data: 价格数据
            pivot_points: 中枢列表
        
        Returns:
            List[Dict]: 第三类卖点列表
        """
        sell_points = []
        
        for pivot in pivot_points:
            try:
                idx = data.index.get_loc(pivot['end'])
                
                search_end = min(idx + 30, len(data))
                
                for j in range(idx + 5, search_end):
                    high_price = data['High'].iloc[j]
                    
                    # 价格接近但未突破中枢上沿（允许5%的误差）
                    if pivot['high'] * 0.95 <= high_price < pivot['high'] * 1.05:
                        sell_point = {
                            'time': data.index[j],
                            'price': high_price,
                            'reason': f"测试中枢上沿阻力（中枢: {pivot['high']:.2f}）",
                            'pivot_high': pivot['high']
                        }
                        sell_points.append(sell_point)
                        
                        logger.debug(f"第三类卖点: {data.index[j]}, 价格: {high_price:.2f}")
                        break
            
            except (KeyError, IndexError):
                continue
        
        return sell_points


# 便捷函数
def identify_signals(
    data: pd.DataFrame,
    pivot_points: List[Dict],
    divergences: List[Dict]
) -> Dict:
    """快速识别买卖点"""
    analyzer = SignalAnalyzer()
    return analyzer.identify_signals(data, pivot_points, divergences)
