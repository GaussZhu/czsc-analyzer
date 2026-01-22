# -*- coding: utf-8 -*-
"""
Pivot Point Identification Module
中枢识别模块 - CZSC理论核心组件
"""

from typing import List, Tuple, Dict
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class PivotAnalyzer:
    """中枢分析器"""
    
    def __init__(self):
        """初始化中枢分析器"""
        logger.debug("中枢分析器初始化完成")
    
    def identify_pivots(self, strokes: List[Tuple]) -> List[Dict]:
        """
        识别中枢
        
        中枢定义：
        - 至少由三笔重叠区间组成
        - 这三笔的价格区间必须有重叠部分
        - 重叠区间即为中枢
        
        Args:
            strokes: 笔列表
        
        Returns:
            List[Dict]: 中枢列表
        """
        if len(strokes) < 3:
            logger.warning("笔数量不足，无法形成中枢（需要至少3笔）")
            return []
        
        pivots = []
        
        try:
            # 检查每三笔是否能形成中枢
            for i in range(len(strokes) - 2):
                stroke1 = strokes[i]
                stroke2 = strokes[i + 1]
                stroke3 = strokes[i + 2]
                
                # 获取每笔的高低点
                high1, low1 = self._get_stroke_range(stroke1)
                high2, low2 = self._get_stroke_range(stroke2)
                high3, low3 = self._get_stroke_range(stroke3)
                
                # 计算三笔的重叠区间
                overlap_high = min(high1, high2, high3)
                overlap_low = max(low1, low2, low3)
                
                # 如果有重叠，则形成中枢
                if overlap_high > overlap_low:
                    pivot = {
                        'start': stroke1[0],      # 开始时间
                        'end': stroke3[1],        # 结束时间
                        'high': overlap_high,     # 中枢上沿
                        'low': overlap_low,       # 中枢下沿
                        'center': (overlap_high + overlap_low) / 2,  # 中枢中心
                        'height': overlap_high - overlap_low,  # 中枢高度
                        'strokes': [stroke1, stroke2, stroke3],  # 组成笔
                        'stroke_count': 3
                    }
                    
                    # 尝试扩展中枢（看后续笔是否也在中枢范围内）
                    pivot = self._extend_pivot(pivot, strokes, i + 3)
                    
                    pivots.append(pivot)
                    
                    logger.debug(
                        f"发现中枢: {pivot['start']} - {pivot['end']}, "
                        f"区间: [{pivot['low']:.2f}, {pivot['high']:.2f}]"
                    )
            
            # 合并重叠的中枢
            pivots = self._merge_overlapping_pivots(pivots)
            
            logger.info(f"中枢识别完成: 共 {len(pivots)} 个中枢")
            return pivots
        
        except Exception as e:
            logger.error(f"中枢识别失败: {str(e)}", exc_info=True)
            return []
    
    def _get_stroke_range(self, stroke: Tuple) -> Tuple[float, float]:
        """
        获取笔的价格区间
        
        Args:
            stroke: 笔元组
        
        Returns:
            Tuple[float, float]: (high, low)
        """
        start_price = stroke[4]
        end_price = stroke[5]
        
        high = max(start_price, end_price)
        low = min(start_price, end_price)
        
        return high, low
    
    def _extend_pivot(self, pivot: Dict, strokes: List[Tuple], start_idx: int) -> Dict:
        """
        扩展中枢（检查后续笔是否也属于中枢）
        
        Args:
            pivot: 当前中枢
            strokes: 所有笔
            start_idx: 开始检查的笔索引
        
        Returns:
            Dict: 扩展后的中枢
        """
        extended = False
        
        for i in range(start_idx, len(strokes)):
            stroke = strokes[i]
            high, low = self._get_stroke_range(stroke)
            
            # 检查笔是否与中枢重叠
            if low < pivot['high'] and high > pivot['low']:
                # 更新中枢范围
                pivot['end'] = stroke[1]
                pivot['strokes'].append(stroke)
                pivot['stroke_count'] += 1
                extended = True
                
                # 重新计算中枢上下沿（取所有笔的重叠区间）
                all_highs = [self._get_stroke_range(s)[0] for s in pivot['strokes']]
                all_lows = [self._get_stroke_range(s)[1] for s in pivot['strokes']]
                
                pivot['high'] = min(all_highs)
                pivot['low'] = max(all_lows)
                pivot['center'] = (pivot['high'] + pivot['low']) / 2
                pivot['height'] = pivot['high'] - pivot['low']
            else:
                # 笔不在中枢范围内，停止扩展
                break
        
        if extended:
            logger.debug(f"中枢扩展: 现有 {pivot['stroke_count']} 笔")
        
        return pivot
    
    def _merge_overlapping_pivots(self, pivots: List[Dict]) -> List[Dict]:
        """
        合并重叠的中枢
        
        Args:
            pivots: 中枢列表
        
        Returns:
            List[Dict]: 合并后的中枢列表
        """
        if len(pivots) <= 1:
            return pivots
        
        merged = []
        current = pivots[0]
        
        for next_pivot in pivots[1:]:
            # 检查时间是否重叠
            if next_pivot['start'] <= current['end']:
                # 合并中枢
                current = {
                    'start': current['start'],
                    'end': max(current['end'], next_pivot['end']),
                    'high': max(current['high'], next_pivot['high']),
                    'low': min(current['low'], next_pivot['low']),
                    'center': (current['center'] + next_pivot['center']) / 2,
                    'height': max(current['height'], next_pivot['height']),
                    'strokes': current['strokes'] + next_pivot['strokes'],
                    'stroke_count': current['stroke_count'] + next_pivot['stroke_count']
                }
            else:
                merged.append(current)
                current = next_pivot
        
        merged.append(current)
        
        return merged
    
    def get_recent_pivots(self, pivots: List[Dict], count: int = 3) -> List[Dict]:
        """
        获取最近的中枢
        
        Args:
            pivots: 所有中枢
            count: 获取数量
        
        Returns:
            List[Dict]: 最近的中枢
        """
        if not pivots:
            return []
        
        # 按结束时间排序
        sorted_pivots = sorted(pivots, key=lambda x: x['end'], reverse=True)
        
        return sorted_pivots[:count]
    
    def analyze_pivot_trend(self, pivots: List[Dict]) -> str:
        """
        分析中枢趋势
        
        Args:
            pivots: 中枢列表
        
        Returns:
            str: 趋势描述 ('up', 'down', 'sideways')
        """
        if len(pivots) < 2:
            return 'unknown'
        
        # 比较最近两个中枢的中心位置
        recent = sorted(pivots, key=lambda x: x['end'], reverse=True)[:2]
        
        latest_center = recent[0]['center']
        previous_center = recent[1]['center']
        
        diff_pct = ((latest_center - previous_center) / previous_center) * 100
        
        if diff_pct > 3:
            return 'up'
        elif diff_pct < -3:
            return 'down'
        else:
            return 'sideways'
    
    def is_price_in_pivot(self, price: float, pivot: Dict) -> bool:
        """
        判断价格是否在中枢内
        
        Args:
            price: 当前价格
            pivot: 中枢
        
        Returns:
            bool: 是否在中枢内
        """
        return pivot['low'] <= price <= pivot['high']
    
    def get_pivot_strength(self, pivot: Dict) -> str:
        """
        评估中枢强度
        
        Args:
            pivot: 中枢
        
        Returns:
            str: 强度评级 ('strong', 'medium', 'weak')
        """
        # 根据笔数量和持续时间评估
        stroke_count = pivot['stroke_count']
        duration = pivot['end'] - pivot['start']
        
        if stroke_count >= 5:
            return 'strong'
        elif stroke_count >= 3:
            return 'medium'
        else:
            return 'weak'


# 便捷函数
def identify_pivots(strokes: List[Tuple]) -> List[Dict]:
    """快速识别中枢"""
    analyzer = PivotAnalyzer()
    return analyzer.identify_pivots(strokes)


def get_recent_pivots(pivots: List[Dict], count: int = 3) -> List[Dict]:
    """快速获取最近的中枢"""
    analyzer = PivotAnalyzer()
    return analyzer.get_recent_pivots(pivots, count)
