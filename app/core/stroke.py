# -*- coding: utf-8 -*-
"""
Stroke Identification Module
笔的识别模块 - CZSC理论核心组件
"""

import pandas as pd
from typing import List, Tuple
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class StrokeAnalyzer:
    """笔分析器"""
    
    def __init__(self):
        """初始化笔分析器"""
        logger.debug("笔分析器初始化完成")
    
    def identify_strokes(self, data: pd.DataFrame) -> List[Tuple]:
        """
        识别笔（从分型点形成笔）
        
        笔的定义：
        - 由相邻的顶分型和底分型连接而成
        - 顶底交替出现
        - 符合CZSC理论的笔的标准
        
        Args:
            data: DataFrame，必须包含 'FractalTop' 和 'FractalBottom' 标记
        
        Returns:
            List[Tuple]: 笔列表，每个元素为 (start_time, end_time, start_type, end_type, start_price, end_price)
        """
        try:
            # 获取所有顶点和底点
            tops = data[data['FractalTop']].index.tolist()
            bottoms = data[data['FractalBottom']].index.tolist()
            
            # 合并并按时间排序
            points = []
            for idx in tops:
                points.append({
                    'time': idx,
                    'type': 'top',
                    'price': data.loc[idx, 'High']
                })
            
            for idx in bottoms:
                points.append({
                    'time': idx,
                    'type': 'bottom',
                    'price': data.loc[idx, 'Low']
                })
            
            # 按时间排序
            points.sort(key=lambda x: x['time'])
            
            if len(points) < 2:
                logger.warning("分型点不足，无法形成笔")
                return []
            
            # 形成笔
            strokes = []
            current = points[0]
            
            for i in range(1, len(points)):
                next_point = points[i]
                
                # 只有类型不同才能形成笔
                if next_point['type'] != current['type']:
                    stroke = (
                        current['time'],      # 起始时间
                        next_point['time'],   # 结束时间
                        current['type'],      # 起始类型
                        next_point['type'],   # 结束类型
                        current['price'],     # 起始价格
                        next_point['price']   # 结束价格
                    )
                    strokes.append(stroke)
                    current = next_point
                else:
                    # 如果类型相同，选择更极端的点
                    if current['type'] == 'top':
                        # 顶点：选择价格更高的
                        if next_point['price'] > current['price']:
                            current = next_point
                    else:
                        # 底点：选择价格更低的
                        if next_point['price'] < current['price']:
                            current = next_point
            
            logger.info(f"笔识别完成: 共 {len(strokes)} 笔")
            return strokes
        
        except Exception as e:
            logger.error(f"笔识别失败: {str(e)}", exc_info=True)
            return []
    
    def get_stroke_info(self, stroke: Tuple) -> dict:
        """
        获取笔的详细信息
        
        Args:
            stroke: 笔元组
        
        Returns:
            dict: 笔的详细信息
        """
        start_time, end_time, start_type, end_type, start_price, end_price = stroke
        
        # 计算涨跌幅
        if start_type == 'bottom' and end_type == 'top':
            direction = 'up'
            change = end_price - start_price
            change_pct = (change / start_price) * 100
        else:
            direction = 'down'
            change = start_price - end_price
            change_pct = (change / start_price) * 100
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'start_type': start_type,
            'end_type': end_type,
            'start_price': start_price,
            'end_price': end_price,
            'direction': direction,
            'change': change,
            'change_pct': change_pct,
            'duration': end_time - start_time
        }
    
    def get_recent_strokes(self, strokes: List[Tuple], count: int = 10) -> List[dict]:
        """
        获取最近的笔
        
        Args:
            strokes: 所有笔的列表
            count: 获取最近的几笔
        
        Returns:
            List[dict]: 最近的笔信息列表
        """
        if not strokes:
            return []
        
        # 取最后几笔
        recent = strokes[-count:] if len(strokes) > count else strokes
        
        return [self.get_stroke_info(stroke) for stroke in recent]
    
    def analyze_stroke_strength(self, stroke: Tuple, data: pd.DataFrame) -> dict:
        """
        分析笔的强度
        
        Args:
            stroke: 笔元组
            data: 价格数据
        
        Returns:
            dict: 强度分析结果
        """
        try:
            start_time, end_time, start_type, end_type, start_price, end_price = stroke
            
            # 获取笔期间的数据
            stroke_data = data.loc[start_time:end_time]
            
            if len(stroke_data) == 0:
                return {'strength': 'unknown'}
            
            # 计算成交量变化
            avg_volume = stroke_data['Volume'].mean()
            
            # 计算价格变化幅度
            price_change_pct = abs((end_price - start_price) / start_price) * 100
            
            # 计算笔的K线数量
            bar_count = len(stroke_data)
            
            # 判断强度
            strength = 'medium'
            if price_change_pct > 10 and bar_count < 10:
                strength = 'strong'
            elif price_change_pct < 3 or bar_count > 20:
                strength = 'weak'
            
            return {
                'strength': strength,
                'price_change_pct': price_change_pct,
                'bar_count': bar_count,
                'avg_volume': avg_volume
            }
        
        except Exception as e:
            logger.error(f"分析笔强度失败: {str(e)}", exc_info=True)
            return {'strength': 'unknown'}
    
    def filter_valid_strokes(self, strokes: List[Tuple], min_change_pct: float = 1.0) -> List[Tuple]:
        """
        过滤有效的笔（根据最小涨跌幅）
        
        Args:
            strokes: 所有笔
            min_change_pct: 最小涨跌幅百分比
        
        Returns:
            List[Tuple]: 有效的笔
        """
        valid_strokes = []
        
        for stroke in strokes:
            start_price = stroke[4]
            end_price = stroke[5]
            
            change_pct = abs((end_price - start_price) / start_price) * 100
            
            if change_pct >= min_change_pct:
                valid_strokes.append(stroke)
        
        logger.info(f"笔过滤完成: 原始 {len(strokes)} 笔，有效 {len(valid_strokes)} 笔")
        return valid_strokes
    
    def get_stroke_direction_sequence(self, strokes: List[Tuple]) -> str:
        """
        获取笔的方向序列
        
        Args:
            strokes: 笔列表
        
        Returns:
            str: 方向序列字符串，如 "↑↓↑↓"
        """
        sequence = []
        for stroke in strokes:
            start_type = stroke[2]
            end_type = stroke[3]
            
            if start_type == 'bottom' and end_type == 'top':
                sequence.append('↑')
            else:
                sequence.append('↓')
        
        return ''.join(sequence)


# 便捷函数
def identify_strokes(data: pd.DataFrame) -> List[Tuple]:
    """快速识别笔"""
    analyzer = StrokeAnalyzer()
    return analyzer.identify_strokes(data)


def get_stroke_info(stroke: Tuple) -> dict:
    """快速获取笔信息"""
    analyzer = StrokeAnalyzer()
    return analyzer.get_stroke_info(stroke)
