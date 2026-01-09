# -*- coding: utf-8 -*-
"""
Fractal Analysis Module
分型识别模块 - CZSC理论核心组件
"""

import pandas as pd
from app.config.settings import get_config
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class FractalAnalyzer:
    """分型分析器"""
    
    def __init__(self, n: int = None):
        """
        初始化分型分析器
        
        Args:
            n: 左右K线数量（默认使用配置值）
        """
        self.config = get_config()
        self.n = n or self.config.fractal.n
        logger.debug(f"分型分析器初始化: n={self.n}")
    
    def identify_fractals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别分型（顶分型和底分型）
        
        分型定义：
        - 顶分型：中间K线的高点高于左右各n根K线的高点
        - 底分型：中间K线的低点低于左右各n根K线的低点
        
        Args:
            data: DataFrame，必须包含 'High' 和 'Low' 列
        
        Returns:
            DataFrame: 添加了 'FractalTop' 和 'FractalBottom' 列的数据
        """
        df = data.copy()
        
        try:
            # 初始化列
            df.loc[:, 'FractalTop'] = False
            df.loc[:, 'FractalBottom'] = False
            
            # 将索引转换为数组以便访问
            indexer = df.index.values
            
            # 识别顶分型
            for i in range(self.n, len(df) - self.n):
                current_high = df['High'].iloc[i]
                
                # 检查左边n根K线
                left_highs = df['High'].iloc[i-self.n:i]
                # 检查右边n根K线
                right_highs = df['High'].iloc[i+1:i+self.n+1]
                
                # 判断是否为顶分型
                if all(current_high > left_highs) and all(current_high > right_highs):
                    df.loc[indexer[i], 'FractalTop'] = True
                    logger.debug(f"发现顶分型: {indexer[i]}, 价格: {current_high:.2f}")
            
            # 识别底分型
            for i in range(self.n, len(df) - self.n):
                current_low = df['Low'].iloc[i]
                
                # 检查左边n根K线
                left_lows = df['Low'].iloc[i-self.n:i]
                # 检查右边n根K线
                right_lows = df['Low'].iloc[i+1:i+self.n+1]
                
                # 判断是否为底分型
                if all(current_low < left_lows) and all(current_low < right_lows):
                    df.loc[indexer[i], 'FractalBottom'] = True
                    logger.debug(f"发现底分型: {indexer[i]}, 价格: {current_low:.2f}")
            
            top_count = df['FractalTop'].sum()
            bottom_count = df['FractalBottom'].sum()
            logger.info(f"分型识别完成: 顶分型={top_count}, 底分型={bottom_count}")
            
            return df
        
        except Exception as e:
            logger.error(f"分型识别失败: {str(e)}", exc_info=True)
            return data
    
    def get_fractal_points(self, data: pd.DataFrame) -> dict:
        """
        获取所有分型点
        
        Args:
            data: 包含分型标记的DataFrame
        
        Returns:
            dict: {'tops': [...], 'bottoms': [...]}
        """
        try:
            tops = data[data['FractalTop']].copy()
            bottoms = data[data['FractalBottom']].copy()
            
            top_points = [
                {
                    'time': idx,
                    'price': row['High'],
                    'type': 'top'
                }
                for idx, row in tops.iterrows()
            ]
            
            bottom_points = [
                {
                    'time': idx,
                    'price': row['Low'],
                    'type': 'bottom'
                }
                for idx, row in bottoms.iterrows()
            ]
            
            logger.debug(f"提取分型点: 顶={len(top_points)}, 底={len(bottom_points)}")
            
            return {
                'tops': top_points,
                'bottoms': bottom_points
            }
        
        except Exception as e:
            logger.error(f"提取分型点失败: {str(e)}", exc_info=True)
            return {'tops': [], 'bottoms': []}
    
    def get_recent_fractals(self, data: pd.DataFrame, count: int = 5) -> dict:
        """
        获取最近的分型点
        
        Args:
            data: 包含分型标记的DataFrame
            count: 获取最近的几个分型
        
        Returns:
            dict: 最近的分型点
        """
        fractals = self.get_fractal_points(data)
        
        # 按时间排序并取最近的
        recent_tops = sorted(fractals['tops'], key=lambda x: x['time'], reverse=True)[:count]
        recent_bottoms = sorted(fractals['bottoms'], key=lambda x: x['time'], reverse=True)[:count]
        
        return {
            'tops': recent_tops,
            'bottoms': recent_bottoms
        }
    
    def is_valid_fractal_sequence(self, data: pd.DataFrame, min_bars: int = 5) -> bool:
        """
        验证分型序列是否有效
        
        Args:
            data: 数据
            min_bars: 分型之间最小K线数量
        
        Returns:
            bool: 序列是否有效
        """
        try:
            fractals = self.get_fractal_points(data)
            all_points = fractals['tops'] + fractals['bottoms']
            
            if len(all_points) < 2:
                return False
            
            # 按时间排序
            all_points.sort(key=lambda x: x['time'])
            
            # 检查相邻分型之间的间隔
            for i in range(1, len(all_points)):
                time_diff = all_points[i]['time'] - all_points[i-1]['time']
                # 这里简化处理，实际应该计算K线数量
                if time_diff < pd.Timedelta(minutes=min_bars * 30):
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"验证分型序列失败: {str(e)}", exc_info=True)
            return False


# 便捷函数
def identify_fractals(data: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """快速识别分型"""
    analyzer = FractalAnalyzer(n)
    return analyzer.identify_fractals(data)


def get_fractal_points(data: pd.DataFrame) -> dict:
    """快速获取分型点"""
    analyzer = FractalAnalyzer()
    return analyzer.get_fractal_points(data)
