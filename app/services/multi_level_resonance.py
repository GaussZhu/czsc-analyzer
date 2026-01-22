# -*- coding: utf-8 -*-
"""
Multi-Level Resonance Analysis Module
多级别共振分析模块 - 实现缠论区间套
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.analyzer import StockAnalyzer
from app.config.logging_config import get_analysis_logger

logger = get_analysis_logger()


class BuyPointType(Enum):
    """买点类型"""
    FIRST = "一买"   # 背驰买点
    SECOND = "二买"  # 回测买点
    THIRD = "三买"   # 突破买点


class TimeLevel(Enum):
    """时间级别"""
    DAILY = "daily"      # 日线
    MIN30 = "30min"      # 30分钟
    MIN5 = "5min"        # 5分钟
    MIN1 = "1min"        # 1分钟（可选）


@dataclass
class BuyPoint:
    """买点数据结构"""
    level: TimeLevel              # 时间级别
    buy_type: BuyPointType       # 买点类型
    price: float                 # 买点价格
    time: str                    # 买点时间
    reason: str                  # 买点理由
    confidence: float            # 置信度 0-1
    support_price: float         # 支撑位
    target_price: float          # 目标位
    stop_loss: float             # 止损位


@dataclass
class ResonanceSignal:
    """共振信号"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    resonance_level: int         # 共振级别 (2-3)
    buy_points: List[BuyPoint]   # 各级别买点列表
    confidence: float            # 综合置信度
    entry_price: float           # 建议入场价
    stop_loss: float             # 止损价
    target_price: float          # 目标价
    expected_profit: float       # 预期收益率 %
    risk_reward: float           # 盈亏比
    analysis_time: str           # 分析时间


class MultiLevelResonanceAnalyzer:
    """多级别共振分析器"""
    
    def __init__(self, use_mysql: bool = True):
        """
        初始化
        
        Args:
            use_mysql: 是否使用MySQL数据源
        """
        self.analyzer = StockAnalyzer(use_mysql=use_mysql)
        
        # 时间窗口容差（用于判断是否在同一时间段出现买点）
        self.time_tolerance = {
            TimeLevel.DAILY: timedelta(days=15),    # 放宽到15天
            TimeLevel.MIN30: timedelta(days=10),    # 放宽到10天
            TimeLevel.MIN5: timedelta(days=5),
        }
    
    def analyze_stock(
        self,
        code: str,
        name: str,
        market: int,
        levels: List[TimeLevel] = None
    ) -> Optional[ResonanceSignal]:
        """
        分析单只股票的多级别共振
        
        Args:
            code: 股票代码
            name: 股票名称
            market: 市场代码
            levels: 要分析的级别列表，默认[日线, 30分钟]
        
        Returns:
            ResonanceSignal 或 None
        """
        if levels is None:
            levels = [TimeLevel.DAILY, TimeLevel.MIN30]  # 只使用支持的级别
        
        logger.info(f"开始多级别共振分析: {code} {name}")
        
        # 收集各级别的买点
        buy_points = []
        
        for level in levels:
            points = self._find_buy_points(code, name, market, level)
            buy_points.extend(points)
        
        if len(buy_points) < 2:
            logger.debug(f"{code} 未发现足够的买点（<2个）")
            return None
        
        # 检查是否形成共振
        resonance_groups = self._check_resonance(buy_points)
        
        if not resonance_groups:
            logger.debug(f"{code} 买点未形成共振")
            return None
        
        # 选择最强的共振信号
        best_resonance = max(resonance_groups, key=lambda x: x.confidence)
        
        logger.info(
            f"✅ {code} 发现{best_resonance.resonance_level}级共振！"
            f"置信度: {best_resonance.confidence:.2f}"
        )
        
        return best_resonance
    
    def _find_buy_points(
        self,
        code: str,
        name: str,
        market: int,
        level: TimeLevel
    ) -> List[BuyPoint]:
        """
        在指定级别寻找买点
        
        Args:
            code: 股票代码
            name: 股票名称
            market: 市场代码
            level: 时间级别
        
        Returns:
            买点列表
        """
        try:
            # 使用StockAnalyzer进行完整分析
            period = level.value
            
            result = self.analyzer.analyze_stock(
                stock_code=code,
                stock_name=name,
                market=market,
                period=period
            )
            
            if not result or 'signals' not in result:
                logger.warning(f"{code} {level.value} 分析失败或无信号")
                return []
            
            signals = result['signals']
            last_price = result['last_price']
            
            # 提取买点
            buy_points = []
            
            # 检查买点（取最近的几个）
            for buy_type in ['first_type', 'second_type', 'third_type']:
                if buy_type not in signals['buy']:
                    continue
                
                # 按时间排序，取最近的买点
                buy_signals = signals['buy'][buy_type]
                if not buy_signals:
                    continue
                
                # 排序并只保留最近的2个
                sorted_signals = sorted(
                    buy_signals,
                    key=lambda x: pd.to_datetime(x.get('time', '1970-01-01')),
                    reverse=True
                )[:2]
                
                for signal in sorted_signals:
                    # 解析买点类型
                    buy_type_enum = self._parse_buy_type_from_key(buy_type)
                    
                    # 获取价格
                    price = float(signal.get('price', last_price))
                    
                    # 获取时间
                    signal_time = signal.get('time')
                    if isinstance(signal_time, pd.Timestamp):
                        time_str = signal_time.strftime('%Y-%m-%d')
                    else:
                        time_str = str(signal_time)
                    
                    # 计算支撑位、目标位、止损位
                    support, target, stop_loss = self._calculate_levels_from_signal(
                        signal, price, level
                    )
                    
                    buy_point = BuyPoint(
                        level=level,
                        buy_type=buy_type_enum,
                        price=price,
                        time=time_str,
                        reason=signal.get('reason', f"{buy_type_enum.value}信号确认"),
                        confidence=self._calculate_confidence_from_type(buy_type_enum, level),
                        support_price=support,
                        target_price=target,
                        stop_loss=stop_loss
                    )
                    
                    buy_points.append(buy_point)
                    logger.debug(
                        f"  {level.value} 发现{buy_type_enum.value}: "
                        f"价格={buy_point.price:.2f}, "
                        f"时间={buy_point.time}"
                    )
            
            return buy_points
        
        except Exception as e:
            logger.error(f"寻找买点时出错 {code} {level.value}: {e}", exc_info=True)
            return []
    
    def _check_resonance(
        self,
        buy_points: List[BuyPoint]
    ) -> List[ResonanceSignal]:
        """
        检查买点是否形成共振
        
        Args:
            buy_points: 所有级别的买点列表
        
        Returns:
            共振信号列表
        """
        if len(buy_points) < 2:
            return []
        
        resonances = []
        
        # 按级别分组
        points_by_level = {}
        for point in buy_points:
            level = point.level
            if level not in points_by_level:
                points_by_level[level] = []
            points_by_level[level].append(point)
        
        # 检查三级共振（最强）
        if len(points_by_level) >= 3:
            three_level = self._find_three_level_resonance(points_by_level)
            if three_level:
                resonances.append(three_level)
        
        # 检查两级共振
        if len(points_by_level) >= 2:
            two_level = self._find_two_level_resonance(points_by_level)
            if two_level:
                resonances.append(two_level)
        
        return resonances
    
    def _find_three_level_resonance(
        self,
        points_by_level: Dict[TimeLevel, List[BuyPoint]]
    ) -> Optional[ResonanceSignal]:
        """寻找三级共振"""
        levels = list(points_by_level.keys())
        
        if len(levels) < 3:
            return None
        
        # 选择日线、30分钟、5分钟
        target_levels = [TimeLevel.DAILY, TimeLevel.MIN30, TimeLevel.MIN5]
        
        if not all(level in points_by_level for level in target_levels):
            return None
        
        # 取每个级别最新的买点
        selected_points = []
        for level in target_levels:
            points = points_by_level[level]
            if points:
                # 选择置信度最高的
                best_point = max(points, key=lambda p: p.confidence)
                selected_points.append(best_point)
        
        if len(selected_points) < 3:
            return None
        
        # 检查时间窗口是否重叠
        if not self._check_time_overlap(selected_points):
            return None
        
        # 构建共振信号
        return self._build_resonance_signal(selected_points, 3)
    
    def _find_two_level_resonance(
        self,
        points_by_level: Dict[TimeLevel, List[BuyPoint]]
    ) -> Optional[ResonanceSignal]:
        """寻找两级共振"""
        levels = list(points_by_level.keys())
        
        if len(levels) < 2:
            return None
        
        # 优先选择：日线+30分钟
        priority_pairs = [
            (TimeLevel.DAILY, TimeLevel.MIN30),
            (TimeLevel.MIN30, TimeLevel.MIN5),
            (TimeLevel.DAILY, TimeLevel.MIN5),
        ]
        
        for level1, level2 in priority_pairs:
            if level1 in points_by_level and level2 in points_by_level:
                points1 = points_by_level[level1]
                points2 = points_by_level[level2]
                
                if points1 and points2:
                    best1 = max(points1, key=lambda p: p.confidence)
                    best2 = max(points2, key=lambda p: p.confidence)
                    
                    selected_points = [best1, best2]
                    
                    if self._check_time_overlap(selected_points):
                        return self._build_resonance_signal(selected_points, 2)
        
        return None
    
    def _check_time_overlap(self, buy_points: List[BuyPoint]) -> bool:
        """检查买点时间是否在容差范围内重叠"""
        if len(buy_points) < 2:
            return False
        
        # 将所有时间转换为datetime
        times = []
        for point in buy_points:
            try:
                # 尝试解析时间（支持多种格式）
                if ' ' in point.time:
                    # 包含时分秒
                    time_obj = datetime.strptime(point.time, '%Y-%m-%d %H:%M:%S')
                else:
                    # 只有日期
                    time_obj = datetime.strptime(point.time, '%Y-%m-%d')
                times.append((time_obj, point.level))
            except:
                logger.warning(f"无法解析时间: {point.time}")
                return False
        
        # 检查是否在容差范围内
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                time1, level1 = times[i]
                time2, level2 = times[j]
                
                # 使用较大级别的容差
                tolerance = max(
                    self.time_tolerance.get(level1, timedelta(days=1)),
                    self.time_tolerance.get(level2, timedelta(days=1))
                )
                
                time_diff = abs(time1 - time2)
                
                if time_diff > tolerance:
                    return False
        
        return True
    
    def _build_resonance_signal(
        self,
        buy_points: List[BuyPoint],
        resonance_level: int
    ) -> ResonanceSignal:
        """构建共振信号"""
        # 综合置信度（加权平均）
        weights = {
            TimeLevel.DAILY: 0.5,    # 日线权重最高
            TimeLevel.MIN30: 0.3,    # 30分钟次之
            TimeLevel.MIN5: 0.2,     # 5分钟最低
        }
        
        total_confidence = 0
        total_weight = 0
        
        for point in buy_points:
            weight = weights.get(point.level, 0.2)
            total_confidence += point.confidence * weight
            total_weight += weight
        
        confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # 共振级别加成
        if resonance_level == 3:
            confidence *= 1.2  # 三级共振加成20%
        
        confidence = min(confidence, 1.0)  # 限制最大值为1.0
        
        # 选择日线级别的买点作为主参考
        daily_point = next(
            (p for p in buy_points if p.level == TimeLevel.DAILY),
            buy_points[0]
        )
        
        # 选择5分钟级别作为精确入场价
        min5_point = next(
            (p for p in buy_points if p.level == TimeLevel.MIN5),
            daily_point
        )
        
        # 计算预期收益和盈亏比
        entry_price = min5_point.price
        target_price = daily_point.target_price
        stop_loss = daily_point.stop_loss
        
        expected_profit = ((target_price - entry_price) / entry_price) * 100
        risk = abs((stop_loss - entry_price) / entry_price) * 100
        risk_reward = expected_profit / risk if risk > 0 else 0
        
        return ResonanceSignal(
            code="",  # 由调用者填充
            name="",  # 由调用者填充
            resonance_level=resonance_level,
            buy_points=buy_points,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            expected_profit=expected_profit,
            risk_reward=risk_reward,
            analysis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _parse_buy_type(self, type_str: str) -> BuyPointType:
        """解析买点类型"""
        if "一买" in type_str:
            return BuyPointType.FIRST
        elif "二买" in type_str:
            return BuyPointType.SECOND
        elif "三买" in type_str:
            return BuyPointType.THIRD
        else:
            return BuyPointType.FIRST
    
    def _parse_buy_type_from_key(self, key: str) -> BuyPointType:
        """从信号key解析买点类型"""
        if "first" in key:
            return BuyPointType.FIRST
        elif "second" in key:
            return BuyPointType.SECOND
        elif "third" in key:
            return BuyPointType.THIRD
        else:
            return BuyPointType.FIRST
    
    def _calculate_levels_from_signal(
        self,
        signal: dict,
        price: float,
        level: TimeLevel
    ) -> Tuple[float, float, float]:
        """
        从信号计算支撑位、目标位、止损位
        
        Returns:
            (support_price, target_price, stop_loss)
        """
        # 根据级别设置比例
        if level == TimeLevel.DAILY:
            support_ratio = 0.05   # 5%
            target_ratio = 0.15    # 15%
            stop_loss_ratio = 0.08 # 8%
        elif level == TimeLevel.MIN30:
            support_ratio = 0.03
            target_ratio = 0.08
            stop_loss_ratio = 0.05
        else:  # MIN5
            support_ratio = 0.02
            target_ratio = 0.05
            stop_loss_ratio = 0.03
        
        # 使用价格计算
        support = price * (1 - support_ratio)
        target = price * (1 + target_ratio)
        stop_loss = price * (1 - stop_loss_ratio)
        
        return support, target, stop_loss
    
    def _calculate_levels(
        self,
        kline: pd.DataFrame,
        signal: dict,
        level: TimeLevel
    ) -> Tuple[float, float, float]:
        """
        计算支撑位、目标位、止损位
        
        Returns:
            (support_price, target_price, stop_loss)
        """
        price = signal['price']
        
        # 根据级别设置比例
        if level == TimeLevel.DAILY:
            support_ratio = 0.05   # 5%
            target_ratio = 0.15    # 15%
            stop_loss_ratio = 0.08 # 8%
        elif level == TimeLevel.MIN30:
            support_ratio = 0.03
            target_ratio = 0.08
            stop_loss_ratio = 0.05
        else:  # MIN5
            support_ratio = 0.02
            target_ratio = 0.05
            stop_loss_ratio = 0.03
        
        # 计算近期低点作为支撑
        recent_low = kline.tail(20)['low'].min()
        support = min(recent_low, price * (1 - support_ratio))
        
        # 计算近期高点作为目标
        recent_high = kline.tail(20)['high'].max()
        target = max(recent_high, price * (1 + target_ratio))
        
        # 止损位
        stop_loss = price * (1 - stop_loss_ratio)
        
        return support, target, stop_loss
    
    def _calculate_confidence(
        self,
        signal: dict,
        level: TimeLevel
    ) -> float:
        """计算买点置信度"""
        base_confidence = {
            '一买': 0.6,
            '二买': 0.8,
            '三买': 0.7,
        }
        
        confidence = base_confidence.get(signal['type'], 0.5)
        
        # 大级别加成
        if level == TimeLevel.DAILY:
            confidence *= 1.1
        elif level == TimeLevel.MIN30:
            confidence *= 1.0
        else:
            confidence *= 0.9
        
        return min(confidence, 1.0)
    
    def _calculate_confidence_from_type(
        self,
        buy_type: BuyPointType,
        level: TimeLevel
    ) -> float:
        """从买点类型计算置信度"""
        base_confidence = {
            BuyPointType.FIRST: 0.6,
            BuyPointType.SECOND: 0.8,
            BuyPointType.THIRD: 0.7,
        }
        
        confidence = base_confidence.get(buy_type, 0.5)
        
        # 大级别加成
        if level == TimeLevel.DAILY:
            confidence *= 1.1
        elif level == TimeLevel.MIN30:
            confidence *= 1.0
        else:
            confidence *= 0.9
        
        return min(confidence, 1.0)
    
    def _get_required_days(self, level: TimeLevel) -> int:
        """获取需要的K线数量"""
        return {
            TimeLevel.DAILY: 300,
            TimeLevel.MIN30: 800,  # 约10天
            TimeLevel.MIN5: 800,   # 约3天
        }.get(level, 300)
    
    def scan_market(
        self,
        stock_list: List[Tuple[str, str, int]],
        min_resonance_level: int = 2
    ) -> List[ResonanceSignal]:
        """
        全市场扫描共振信号
        
        Args:
            stock_list: [(code, name, market), ...]
            min_resonance_level: 最小共振级别（2或3）
        
        Returns:
            共振信号列表（按置信度降序）
        """
        logger.info(f"开始全市场扫描，共{len(stock_list)}只股票")
        
        resonances = []
        
        for code, name, market in stock_list:
            try:
                signal = self.analyze_stock(code, name, market)
                
                if signal and signal.resonance_level >= min_resonance_level:
                    signal.code = code
                    signal.name = name
                    resonances.append(signal)
                    
                    logger.info(
                        f"✅ {code} {name} - "
                        f"{signal.resonance_level}级共振 "
                        f"置信度:{signal.confidence:.2f} "
                        f"预期收益:{signal.expected_profit:.1f}%"
                    )
            
            except Exception as e:
                logger.error(f"分析 {code} 时出错: {e}")
                continue
        
        # 按置信度降序排序
        resonances.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"扫描完成，发现{len(resonances)}个共振信号")
        
        return resonances
    
    def close(self):
        """关闭数据源连接"""
        # UnifiedDataSource 会自动管理连接，不需要手动关闭
        pass


# ========== 辅助函数 ==========

def format_resonance_report(signal: ResonanceSignal) -> str:
    """格式化共振信号报告"""
    report = f"""
{'='*60}
【{signal.resonance_level}级共振信号】{signal.code} {signal.name}
{'='*60}

📊 综合评估:
  • 共振级别: {signal.resonance_level}级
  • 置信度: {signal.confidence:.1%}
  • 预期收益: +{signal.expected_profit:.1f}%
  • 盈亏比: {signal.risk_reward:.2f}:1

💰 操作建议:
  • 建议入场价: ¥{signal.entry_price:.2f}
  • 目标价: ¥{signal.target_price:.2f}
  • 止损价: ¥{signal.stop_loss:.2f}

🎯 各级别买点:
"""
    
    for point in signal.buy_points:
        report += f"""
  [{point.level.value}] {point.buy_type.value}
    • 价格: ¥{point.price:.2f}
    • 时间: {point.time}
    • 理由: {point.reason}
    • 置信度: {point.confidence:.1%}
    • 支撑位: ¥{point.support_price:.2f}
    • 目标位: ¥{point.target_price:.2f}
"""
    
    report += f"""
⏰ 分析时间: {signal.analysis_time}

{'='*60}
"""
    
    return report


if __name__ == '__main__':
    # 测试代码
    analyzer = MultiLevelResonanceAnalyzer(use_mysql=True)
    
    # 测试单只股票
    signal = analyzer.analyze_stock(
        code='600000',
        name='浦发银行',
        market=1
    )
    
    if signal:
        print(format_resonance_report(signal))
    else:
        print("未发现共振信号")
    
    analyzer.close()
