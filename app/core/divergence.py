# -*- coding: utf-8 -*-
"""
Divergence Detection Module
背驰检测模块 - CZSC理论核心组件
"""

import pandas as pd
from typing import List, Tuple, Dict
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class DivergenceAnalyzer:
    """背驰分析器"""
    
    def __init__(self):
        """初始化背驰分析器"""
        logger.debug("背驰分析器初始化完成")
    
    def detect_divergences(self, data: pd.DataFrame, strokes: List[Tuple]) -> List[Dict]:
        """
        检测背驰
        
        背驰定义：
        - 价格创新高/新低，但MACD指标未创新高/新低
        - 常见于趋势末端，预示反转
        
        Args:
            data: 包含MACD数据的DataFrame
            strokes: 笔列表
        
        Returns:
            List[Dict]: 背驰点列表
        """
        if len(strokes) < 4:
            logger.warning("笔数量不足，无法检测背驰（需要至少4笔）")
            return []
        
        divergences = []
        
        try:
            # 分离上升笔和下降笔
            up_strokes = [s for s in strokes if s[2] == 'bottom' and s[3] == 'top']
            down_strokes = [s for s in strokes if s[2] == 'top' and s[3] == 'bottom']
            
            # 检测看涨背驰（价格创新低，但MACD上升）
            bullish_divs = self._detect_bullish_divergence(data, down_strokes)
            divergences.extend(bullish_divs)
            
            # 检测看跌背驰（价格创新高，但MACD下降）
            bearish_divs = self._detect_bearish_divergence(data, up_strokes)
            divergences.extend(bearish_divs)
            
            logger.info(f"背驰检测完成: 看涨={len(bullish_divs)}, 看跌={len(bearish_divs)}")
            return divergences
        
        except Exception as e:
            logger.error(f"背驰检测失败: {str(e)}", exc_info=True)
            return []
    
    def _detect_bullish_divergence(self, data: pd.DataFrame, down_strokes: List[Tuple]) -> List[Dict]:
        """
        检测看涨背驰（底背驰）
        
        Args:
            data: MACD数据
            down_strokes: 下降笔列表
        
        Returns:
            List[Dict]: 看涨背驰列表
        """
        divergences = []
        
        for i in range(1, len(down_strokes)):
            current = down_strokes[i]
            previous = down_strokes[i - 1]
            
            # 当前笔的结束价格（低点）
            current_price = current[5]
            previous_price = previous[5]
            
            # 价格创新低
            if current_price < previous_price:
                try:
                    # 获取MACD柱状图值
                    current_macd = data.loc[current[1], 'Histogram']
                    previous_macd = data.loc[previous[1], 'Histogram']
                    
                    # MACD柱状图上升（更高的低点）
                    if current_macd > previous_macd and current_macd < 0:
                        divergence = {
                            'type': 'bullish',
                            'time': current[1],
                            'price': current_price,
                            'previous_time': previous[1],
                            'previous_price': previous_price,
                            'macd': current_macd,
                            'previous_macd': previous_macd,
                            'strength': self._calculate_divergence_strength(
                                previous_price, current_price,
                                previous_macd, current_macd
                            )
                        }
                        divergences.append(divergence)
                        
                        logger.debug(
                            f"发现看涨背驰: {current[1]}, "
                            f"价格: {previous_price:.2f} -> {current_price:.2f}, "
                            f"MACD: {previous_macd:.4f} -> {current_macd:.4f}"
                        )
                except KeyError:
                    # 时间点不在数据中
                    continue
        
        return divergences
    
    def _detect_bearish_divergence(self, data: pd.DataFrame, up_strokes: List[Tuple]) -> List[Dict]:
        """
        检测看跌背驰（顶背驰）
        
        Args:
            data: MACD数据
            up_strokes: 上升笔列表
        
        Returns:
            List[Dict]: 看跌背驰列表
        """
        divergences = []
        
        for i in range(1, len(up_strokes)):
            current = up_strokes[i]
            previous = up_strokes[i - 1]
            
            # 当前笔的结束价格（高点）
            current_price = current[5]
            previous_price = previous[5]
            
            # 价格创新高
            if current_price > previous_price:
                try:
                    # 获取MACD柱状图值
                    current_macd = data.loc[current[1], 'Histogram']
                    previous_macd = data.loc[previous[1], 'Histogram']
                    
                    # MACD柱状图下降（更低的高点）
                    if current_macd < previous_macd and current_macd > 0:
                        divergence = {
                            'type': 'bearish',
                            'time': current[1],
                            'price': current_price,
                            'previous_time': previous[1],
                            'previous_price': previous_price,
                            'macd': current_macd,
                            'previous_macd': previous_macd,
                            'strength': self._calculate_divergence_strength(
                                previous_price, current_price,
                                previous_macd, current_macd
                            )
                        }
                        divergences.append(divergence)
                        
                        logger.debug(
                            f"发现看跌背驰: {current[1]}, "
                            f"价格: {previous_price:.2f} -> {current_price:.2f}, "
                            f"MACD: {previous_macd:.4f} -> {current_macd:.4f}"
                        )
                except KeyError:
                    # 时间点不在数据中
                    continue
        
        return divergences
    
    def _calculate_divergence_strength(
        self,
        prev_price: float,
        curr_price: float,
        prev_macd: float,
        curr_macd: float
    ) -> str:
        """
        计算背驰强度
        
        Args:
            prev_price: 前一个价格
            curr_price: 当前价格
            prev_macd: 前一个MACD值
            curr_macd: 当前MACD值
        
        Returns:
            str: 强度等级 ('strong', 'medium', 'weak')
        """
        # 计算价格变化和MACD变化的背离程度
        price_change = abs(curr_price - prev_price) / prev_price * 100
        macd_change = abs(curr_macd - prev_macd)
        
        # 背离越明显，强度越大
        if price_change > 5 and macd_change > 0.5:
            return 'strong'
        elif price_change > 2 and macd_change > 0.2:
            return 'medium'
        else:
            return 'weak'
    
    def get_recent_divergences(self, divergences: List[Dict], count: int = 5) -> List[Dict]:
        """
        获取最近的背驰
        
        Args:
            divergences: 所有背驰
            count: 获取数量
        
        Returns:
            List[Dict]: 最近的背驰
        """
        if not divergences:
            return []
        
        # 按时间排序
        sorted_divs = sorted(divergences, key=lambda x: x['time'], reverse=True)
        
        return sorted_divs[:count]
    
    def filter_strong_divergences(self, divergences: List[Dict]) -> List[Dict]:
        """
        筛选强背驰
        
        Args:
            divergences: 所有背驰
        
        Returns:
            List[Dict]: 强背驰列表
        """
        return [d for d in divergences if d.get('strength') == 'strong']
    
    def has_recent_divergence(
        self,
        divergences: List[Dict],
        div_type: str,
        days: int = 3
    ) -> bool:
        """
        检查最近是否有背驰
        
        Args:
            divergences: 背驰列表
            div_type: 背驰类型 ('bullish' 或 'bearish')
            days: 最近几天
        
        Returns:
            bool: 是否存在
        """
        import datetime
        
        recent_time = pd.Timestamp.now() - pd.Timedelta(days=days)
        
        for div in divergences:
            if div['type'] == div_type and div['time'] > recent_time:
                return True
        
        return False


# 便捷函数
def detect_divergences(data: pd.DataFrame, strokes: List[Tuple]) -> List[Dict]:
    """快速检测背驰"""
    analyzer = DivergenceAnalyzer()
    return analyzer.detect_divergences(data, strokes)


def get_recent_divergences(divergences: List[Dict], count: int = 5) -> List[Dict]:
    """快速获取最近的背驰"""
    analyzer = DivergenceAnalyzer()
    return analyzer.get_recent_divergences(divergences, count)
