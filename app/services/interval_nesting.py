# -*- coding: utf-8 -*-
"""
缠论区间套系统 v2.0 - 从大到小逐级确认
策略：周线确认趋势 → 日线找买点 → 30分钟/5分钟精确入场
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.analyzer import StockAnalyzer
from app.core.data_source import UnifiedDataSource, DataSourceType
from app.config.logging_config import get_analysis_logger

logger = get_analysis_logger()


class TrendDirection(Enum):
    """趋势方向"""
    UP = "上涨"
    DOWN = "下跌"
    SIDEWAYS = "震荡"
    UNKNOWN = "未知"


class BuyPointType(Enum):
    """买点类型"""
    FIRST = "一买"
    SECOND = "二买"
    THIRD = "三买"


@dataclass
class PriceZone:
    """价格区间"""
    low: float           # 区间下沿
    high: float          # 区间上沿
    optimal: float       # 最优入场价
    reason: str          # 理由


@dataclass
class SignalLevel:
    """单个级别的信号"""
    level: str                          # 级别名称（weekly/daily/30min/5min）
    has_signal: bool                    # 是否有信号
    signal_type: Optional[BuyPointType] # 信号类型
    price: Optional[float]              # 信号价格
    time: Optional[str]                 # 信号时间
    trend: TrendDirection               # 当前趋势
    reason: str                         # 原因说明


@dataclass
class IntervalNestingSignal:
    """区间套信号"""
    code: str                           # 股票代码
    name: str                           # 股票名称
    
    # 各级别信号
    weekly_signal: SignalLevel          # 周线
    daily_signal: SignalLevel           # 日线（核心）
    min30_signal: Optional[SignalLevel] # 30分钟
    min5_signal: Optional[SignalLevel]  # 5分钟
    
    # 综合判断
    is_valid: bool                      # 是否有效信号
    confidence: float                   # 置信度 0-1
    
    # 操作建议
    entry_zone: PriceZone              # 入场价格区间
    stop_loss: float                   # 止损价
    target_price: float                # 目标价
    expected_profit: float             # 预期收益率 %
    risk_reward: float                 # 盈亏比
    
    # 分析说明
    summary: str                       # 综合评价
    analysis_time: str                 # 分析时间


class IntervalNestingAnalyzer:
    """区间套分析器 - 从大到小逐级确认"""
    
    def __init__(self, use_mysql: bool = True):
        """
        初始化
        
        Args:
            use_mysql: 是否使用MySQL数据源
        """
        self.analyzer = StockAnalyzer(use_mysql=use_mysql)
        
        # 设置数据源类型
        source_type = DataSourceType.MYSQL if use_mysql else DataSourceType.TDX
        self.data_source = UnifiedDataSource(source_type=source_type)
    
    def analyze(
        self,
        code: str,
        name: str,
        market: int,
        check_intraday: bool = True
    ) -> Optional[IntervalNestingSignal]:
        """
        完整的区间套分析
        
        Args:
            code: 股票代码
            name: 股票名称
            market: 市场代码
            check_intraday: 是否检查日内级别（30分钟/5分钟）
        
        Returns:
            IntervalNestingSignal 或 None
        """
        logger.info(f"🎯 开始区间套分析: {code} {name}")
        
        try:
            # Step 1: 分析周线（确认大趋势）
            logger.info("  📊 Step 1: 检查周线趋势...")
            weekly_signal = self._analyze_weekly(code, name, market)
            
            if weekly_signal.trend == TrendDirection.DOWN:
                logger.warning(f"  ❌ {code} 周线处于下跌趋势，不建议操作")
                return None
            
            logger.info(f"  ✓ 周线趋势: {weekly_signal.trend.value}")
            
            # Step 2: 分析日线（寻找买点）
            logger.info("  📊 Step 2: 检查日线买点...")
            daily_signal = self._analyze_daily(code, name, market)
            
            if not daily_signal.has_signal:
                logger.debug(f"  ⚠ {code} 日线未发现买点")
                return None
            
            logger.info(f"  ✓ 日线发现: {daily_signal.signal_type.value} @ ¥{daily_signal.price:.2f}")
            
            # Step 3: 分析30分钟和5分钟（精确入场点）
            min30_signal = None
            min5_signal = None
            
            if check_intraday:
                logger.info("  📊 Step 3: 寻找精确入场点...")
                
                # 30分钟
                min30_signal = self._analyze_intraday(code, name, market, '30min')
                if min30_signal and min30_signal.has_signal:
                    logger.info(f"  ✓ 30分钟: {min30_signal.signal_type.value} @ ¥{min30_signal.price:.2f}")
                
                # 5分钟（实时获取）
                min5_signal = self._analyze_intraday_realtime(code, name, market, '5min')
                if min5_signal and min5_signal.has_signal:
                    logger.info(f"  ✓ 5分钟: {min5_signal.signal_type.value} @ ¥{min5_signal.price:.2f}")
            
            # Step 4: 综合判断和生成信号
            logger.info("  📊 Step 4: 生成综合信号...")
            signal = self._build_signal(
                code, name,
                weekly_signal, daily_signal,
                min30_signal, min5_signal
            )
            
            if signal and signal.is_valid:
                logger.info(
                    f"  ✅ {code} 区间套信号确认！"
                    f"置信度: {signal.confidence:.0%}, "
                    f"入场区间: ¥{signal.entry_zone.low:.2f}-{signal.entry_zone.high:.2f}"
                )
            
            return signal
        
        except Exception as e:
            logger.error(f"区间套分析失败 {code}: {e}", exc_info=True)
            return None
    
    def _analyze_weekly(
        self,
        code: str,
        name: str,
        market: int
    ) -> SignalLevel:
        """
        分析周线趋势
        
        策略：
        1. 检查是否处于上涨段（价格在中枢上方）
        2. 检查是否有顶背驰（需要警惕）
        """
        try:
            # 尝试从MySQL获取周线数据
            weekly_data = self.data_source.get_stock_data(
                code, market, 'weekly', prefer_mysql=True
            )
            
            if weekly_data is None or len(weekly_data) < 30:
                # 如果没有周线数据，从日线推导
                logger.warning(f"{code} 周线数据不足，使用日线推导")
                return self._infer_weekly_from_daily(code, name, market)
            
            # 简单判断趋势
            recent_prices = weekly_data['Close'].tail(20)
            current_price = recent_prices.iloc[-1]
            ma20 = recent_prices.mean()
            
            # 判断趋势
            if current_price > ma20 * 1.05:
                trend = TrendDirection.UP
                reason = f"价格({current_price:.2f})在20周均线({ma20:.2f})上方5%以上"
            elif current_price < ma20 * 0.95:
                trend = TrendDirection.DOWN
                reason = f"价格({current_price:.2f})在20周均线({ma20:.2f})下方5%以上"
            else:
                trend = TrendDirection.SIDEWAYS
                reason = f"价格({current_price:.2f})在20周均线({ma20:.2f})附近震荡"
            
            return SignalLevel(
                level='weekly',
                has_signal=False,  # 周线不用找买点，只看趋势
                signal_type=None,
                price=current_price,
                time=str(weekly_data.index[-1]),
                trend=trend,
                reason=reason
            )
        
        except Exception as e:
            logger.error(f"周线分析失败: {e}")
            # 返回未知状态
            return SignalLevel(
                level='weekly',
                has_signal=False,
                signal_type=None,
                price=None,
                time=None,
                trend=TrendDirection.UNKNOWN,
                reason="周线数据获取失败"
            )
    
    def _infer_weekly_from_daily(
        self,
        code: str,
        name: str,
        market: int
    ) -> SignalLevel:
        """从日线数据推导周线趋势"""
        try:
            daily_data = self.data_source.get_stock_data(
                code, market, 'daily', prefer_mysql=True
            )
            
            if daily_data is None or len(daily_data) < 100:
                return SignalLevel(
                    level='weekly',
                    has_signal=False,
                    signal_type=None,
                    price=None,
                    time=None,
                    trend=TrendDirection.UNKNOWN,
                    reason="数据不足"
                )
            
            # 使用60日均线代替周线判断
            recent_prices = daily_data['Close'].tail(100)
            current_price = recent_prices.iloc[-1]
            ma60 = recent_prices.tail(60).mean()
            
            if current_price > ma60 * 1.1:
                trend = TrendDirection.UP
                reason = f"日线60日均线({ma60:.2f})之上"
            elif current_price < ma60 * 0.9:
                trend = TrendDirection.DOWN
                reason = f"日线60日均线({ma60:.2f})之下"
            else:
                trend = TrendDirection.SIDEWAYS
                reason = f"日线60日均线({ma60:.2f})附近"
            
            return SignalLevel(
                level='weekly',
                has_signal=False,
                signal_type=None,
                price=current_price,
                time=str(daily_data.index[-1]),
                trend=trend,
                reason=reason
            )
        
        except Exception as e:
            logger.error(f"从日线推导周线失败: {e}")
            return SignalLevel(
                level='weekly',
                has_signal=False,
                signal_type=None,
                price=None,
                time=None,
                trend=TrendDirection.UNKNOWN,
                reason="推导失败"
            )
    
    def _analyze_daily(
        self,
        code: str,
        name: str,
        market: int
    ) -> SignalLevel:
        """
        分析日线买点（核心）
        """
        try:
            # 使用StockAnalyzer进行完整分析
            result = self.analyzer.analyze_stock(
                stock_code=code,
                stock_name=name,
                market=market,
                period='daily'
            )
            
            if not result or 'signals' not in result:
                return SignalLevel(
                    level='daily',
                    has_signal=False,
                    signal_type=None,
                    price=result.get('last_price') if result else None,
                    time=None,
                    trend=TrendDirection.UNKNOWN,
                    reason="分析失败"
                )
            
            signals = result['signals']
            last_price = result['last_price']
            
            # 查找最近的买点（优先级：二买 > 三买 > 一买）
            buy_priority = [
                ('second_type', BuyPointType.SECOND),
                ('third_type', BuyPointType.THIRD),
                ('first_type', BuyPointType.FIRST),
            ]
            
            for key, buy_type in buy_priority:
                if key in signals['buy'] and signals['buy'][key]:
                    # 取最近的信号
                    latest_signal = max(
                        signals['buy'][key],
                        key=lambda x: pd.to_datetime(x.get('time', '1970-01-01'))
                    )
                    
                    signal_time = latest_signal.get('time')
                    if isinstance(signal_time, pd.Timestamp):
                        time_str = signal_time.strftime('%Y-%m-%d')
                    else:
                        time_str = str(signal_time)
                    
                    return SignalLevel(
                        level='daily',
                        has_signal=True,
                        signal_type=buy_type,
                        price=float(latest_signal.get('price', last_price)),
                        time=time_str,
                        trend=TrendDirection.UP,  # 有买点说明趋势向上
                        reason=latest_signal.get('reason', f'{buy_type.value}信号确认')
                    )
            
            # 没有找到买点
            return SignalLevel(
                level='daily',
                has_signal=False,
                signal_type=None,
                price=last_price,
                time=str(result.get('last_update', '')),
                trend=TrendDirection.SIDEWAYS,
                reason="未发现买点信号"
            )
        
        except Exception as e:
            logger.error(f"日线分析失败: {e}", exc_info=True)
            return SignalLevel(
                level='daily',
                has_signal=False,
                signal_type=None,
                price=None,
                time=None,
                trend=TrendDirection.UNKNOWN,
                reason=f"分析失败: {str(e)}"
            )
    
    def _analyze_intraday(
        self,
        code: str,
        name: str,
        market: int,
        period: str
    ) -> Optional[SignalLevel]:
        """
        分析日内级别（30分钟）
        """
        try:
            result = self.analyzer.analyze_stock(
                stock_code=code,
                stock_name=name,
                market=market,
                period=period
            )
            
            if not result or 'signals' not in result:
                return None
            
            signals = result['signals']
            last_price = result['last_price']
            
            # 查找买点
            for key in ['third_type', 'second_type', 'first_type']:
                if key in signals['buy'] and signals['buy'][key]:
                    latest_signal = max(
                        signals['buy'][key],
                        key=lambda x: pd.to_datetime(x.get('time', '1970-01-01'))
                    )
                    
                    buy_type = {
                        'first_type': BuyPointType.FIRST,
                        'second_type': BuyPointType.SECOND,
                        'third_type': BuyPointType.THIRD,
                    }[key]
                    
                    signal_time = latest_signal.get('time')
                    if isinstance(signal_time, pd.Timestamp):
                        time_str = signal_time.strftime('%Y-%m-%d %H:%M')
                    else:
                        time_str = str(signal_time)
                    
                    return SignalLevel(
                        level=period,
                        has_signal=True,
                        signal_type=buy_type,
                        price=float(latest_signal.get('price', last_price)),
                        time=time_str,
                        trend=TrendDirection.UP,
                        reason=latest_signal.get('reason', f'{buy_type.value}')
                    )
            
            return None
        
        except Exception as e:
            logger.error(f"{period}分析失败: {e}")
            return None
    
    def _analyze_intraday_realtime(
        self,
        code: str,
        name: str,
        market: int,
        period: str
    ) -> Optional[SignalLevel]:
        """
        实时获取5分钟数据分析（从TDX）
        """
        try:
            # 实时从TDX获取5分钟数据
            from czsc.connectors import cooperation as coo
            
            # 使用pytdx实时获取
            logger.info(f"  实时获取{code}的5分钟数据...")
            
            # TODO: 实现实时5分钟数据获取
            # 由于时间关系，这里先返回None
            # 实际使用时可以调用pytdx的get_security_bars获取实时数据
            
            return None
        
        except Exception as e:
            logger.error(f"5分钟实时分析失败: {e}")
            return None
    
    def _build_signal(
        self,
        code: str,
        name: str,
        weekly_signal: SignalLevel,
        daily_signal: SignalLevel,
        min30_signal: Optional[SignalLevel],
        min5_signal: Optional[SignalLevel]
    ) -> Optional[IntervalNestingSignal]:
        """
        综合各级别信号，构建最终信号
        """
        # 必要条件：日线有买点
        if not daily_signal.has_signal:
            return None
        
        # 过滤条件：周线不能是明确下跌
        if weekly_signal.trend == TrendDirection.DOWN:
            return None
        
        # 计算置信度
        confidence = 0.5  # 基础置信度
        
        # 日线买点加分
        if daily_signal.signal_type == BuyPointType.SECOND:
            confidence += 0.3  # 二买最可靠
        elif daily_signal.signal_type == BuyPointType.THIRD:
            confidence += 0.2  # 三买次之
        else:
            confidence += 0.1  # 一买风险较大
        
        # 周线趋势加分
        if weekly_signal.trend == TrendDirection.UP:
            confidence += 0.2  # 周线上涨加分
        elif weekly_signal.trend == TrendDirection.SIDEWAYS:
            confidence += 0.1  # 周线震荡略加分
        
        # 30分钟共振加分
        if min30_signal and min30_signal.has_signal:
            confidence += 0.1
        
        # 5分钟共振加分
        if min5_signal and min5_signal.has_signal:
            confidence += 0.05
        
        confidence = min(confidence, 0.95)  # 最高95%
        
        # 构建入场价格区间
        entry_zone = self._calculate_entry_zone(
            daily_signal, min30_signal, min5_signal
        )
        
        # 计算止损和目标
        stop_loss, target = self._calculate_targets(
            daily_signal, entry_zone.optimal
        )
        
        # 计算预期收益
        expected_profit = ((target - entry_zone.optimal) / entry_zone.optimal) * 100
        risk = abs((stop_loss - entry_zone.optimal) / entry_zone.optimal) * 100
        risk_reward = expected_profit / risk if risk > 0 else 0
        
        # 生成综合评价
        summary = self._generate_summary(
            daily_signal, weekly_signal, min30_signal, confidence
        )
        
        return IntervalNestingSignal(
            code=code,
            name=name,
            weekly_signal=weekly_signal,
            daily_signal=daily_signal,
            min30_signal=min30_signal,
            min5_signal=min5_signal,
            is_valid=True,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            target_price=target,
            expected_profit=expected_profit,
            risk_reward=risk_reward,
            summary=summary,
            analysis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _calculate_entry_zone(
        self,
        daily_signal: SignalLevel,
        min30_signal: Optional[SignalLevel],
        min5_signal: Optional[SignalLevel]
    ) -> PriceZone:
        """计算入场价格区间"""
        daily_price = daily_signal.price
        
        # 如果有30分钟或5分钟信号，以小级别为准
        if min5_signal and min5_signal.has_signal:
            optimal = min5_signal.price
            low = optimal * 0.98
            high = optimal * 1.02
            reason = f"5分钟{min5_signal.signal_type.value}附近"
        elif min30_signal and min30_signal.has_signal:
            optimal = min30_signal.price
            low = optimal * 0.98
            high = optimal * 1.03
            reason = f"30分钟{min30_signal.signal_type.value}附近"
        else:
            # 只有日线信号，设置一个合理区间
            optimal = daily_price
            low = optimal * 0.95
            high = optimal * 1.02
            reason = f"日线{daily_signal.signal_type.value}附近"
        
        return PriceZone(
            low=low,
            high=high,
            optimal=optimal,
            reason=reason
        )
    
    def _calculate_targets(
        self,
        daily_signal: SignalLevel,
        entry_price: float
    ) -> Tuple[float, float]:
        """计算止损和目标价"""
        # 根据买点类型设置不同的止损和目标
        if daily_signal.signal_type == BuyPointType.FIRST:
            stop_loss_ratio = 0.08  # 一买止损8%
            target_ratio = 0.20     # 目标20%
        elif daily_signal.signal_type == BuyPointType.SECOND:
            stop_loss_ratio = 0.05  # 二买止损5%
            target_ratio = 0.15     # 目标15%
        else:  # THIRD
            stop_loss_ratio = 0.05  # 三买止损5%
            target_ratio = 0.12     # 目标12%
        
        stop_loss = entry_price * (1 - stop_loss_ratio)
        target = entry_price * (1 + target_ratio)
        
        return stop_loss, target
    
    def _generate_summary(
        self,
        daily_signal: SignalLevel,
        weekly_signal: SignalLevel,
        min30_signal: Optional[SignalLevel],
        confidence: float
    ) -> str:
        """生成综合评价"""
        parts = []
        
        # 周线趋势
        parts.append(f"周线{weekly_signal.trend.value}")
        
        # 日线买点
        parts.append(f"日线{daily_signal.signal_type.value}")
        
        # 30分钟共振
        if min30_signal and min30_signal.has_signal:
            parts.append(f"30分钟{min30_signal.signal_type.value}共振")
        
        # 置信度评价
        if confidence >= 0.8:
            rating = "强烈推荐"
        elif confidence >= 0.7:
            rating = "推荐"
        elif confidence >= 0.6:
            rating = "可关注"
        else:
            rating = "谨慎"
        
        return f"{', '.join(parts)}，综合评级: {rating}"
    
    def scan_market(
        self,
        stock_list: List[Tuple[str, str, int]],
        min_confidence: float = 0.6
    ) -> List[IntervalNestingSignal]:
        """
        全市场扫描
        
        Args:
            stock_list: [(code, name, market), ...]
            min_confidence: 最小置信度
        
        Returns:
            信号列表（按置信度降序）
        """
        logger.info(f"开始全市场区间套扫描，共{len(stock_list)}只股票")
        
        signals = []
        
        for code, name, market in stock_list:
            try:
                signal = self.analyze(code, name, market, check_intraday=True)
                
                if signal and signal.confidence >= min_confidence:
                    signals.append(signal)
                    
                    logger.info(
                        f"✅ {code} {name} - "
                        f"置信度:{signal.confidence:.0%} "
                        f"{signal.summary}"
                    )
            
            except Exception as e:
                logger.error(f"扫描 {code} 时出错: {e}")
                continue
        
        # 按置信度降序排序
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"扫描完成，发现{len(signals)}个有效信号")
        
        return signals
    
    def close(self):
        """关闭连接"""
        pass


# ========== 辅助函数 ==========

def format_signal_report(signal: IntervalNestingSignal) -> str:
    """格式化信号报告"""
    report = f"""
{'='*70}
【区间套信号】{signal.code} {signal.name}
{'='*70}

📊 综合评估:
  {signal.summary}
  置信度: {signal.confidence:.0%} {'🔥' if signal.confidence >= 0.8 else '✓'}

🎯 各级别状态:
  
  【周线】{signal.weekly_signal.trend.value}
    {signal.weekly_signal.reason}
  
  【日线】{signal.daily_signal.signal_type.value if signal.daily_signal.has_signal else '无信号'}
    价格: ¥{signal.daily_signal.price:.2f}
    时间: {signal.daily_signal.time}
    理由: {signal.daily_signal.reason}
"""
    
    if signal.min30_signal and signal.min30_signal.has_signal:
        report += f"""  
  【30分钟】{signal.min30_signal.signal_type.value}
    价格: ¥{signal.min30_signal.price:.2f}
    时间: {signal.min30_signal.time}
"""
    
    if signal.min5_signal and signal.min5_signal.has_signal:
        report += f"""  
  【5分钟】{signal.min5_signal.signal_type.value}
    价格: ¥{signal.min5_signal.price:.2f}
    时间: {signal.min5_signal.time}
"""
    
    report += f"""
💰 操作建议:
  
  入场区间: ¥{signal.entry_zone.low:.2f} - ¥{signal.entry_zone.high:.2f}
  最优价格: ¥{signal.entry_zone.optimal:.2f}
  理由: {signal.entry_zone.reason}
  
  止损价: ¥{signal.stop_loss:.2f} (风险: {abs((signal.stop_loss - signal.entry_zone.optimal) / signal.entry_zone.optimal * 100):.1f}%)
  目标价: ¥{signal.target_price:.2f} (收益: +{signal.expected_profit:.1f}%)
  盈亏比: {signal.risk_reward:.2f}:1

⏰ 分析时间: {signal.analysis_time}

{'='*70}
"""
    
    return report


if __name__ == '__main__':
    # 测试代码
    analyzer = IntervalNestingAnalyzer(use_mysql=True)
    
    # 测试单只股票
    signal = analyzer.analyze(
        code='600000',
        name='浦发银行',
        market=1,
        check_intraday=True
    )
    
    if signal:
        print(format_signal_report(signal))
    else:
        print("未发现有效信号")
    
    analyzer.close()
