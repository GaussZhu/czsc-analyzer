# -*- coding: utf-8 -*-
"""
缠论区间套分析系统 v2.0
核心策略：
1. 日线找买点（主要信号）
2. 周线验证大趋势（是否在上涨段）
3. 30分钟/5分钟找精确入场区间
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.analyzer import StockAnalyzer
from app.config.logging_config import get_analysis_logger

logger = get_analysis_logger()


class TrendDirection(Enum):
    """趋势方向"""
    UP = "上涨段"
    DOWN = "下跌段"
    SIDEWAYS = "盘整"
    UNKNOWN = "未知"


class BuyPointType(Enum):
    """买点类型"""
    FIRST = "一买"
    SECOND = "二买"
    THIRD = "三买"


@dataclass
class DailyBuyPoint:
    """日线买点"""
    code: str
    name: str
    buy_type: BuyPointType
    price: float
    time: str
    reason: str
    confidence: float


@dataclass
class TrendValidation:
    """趋势验证结果"""
    weekly_trend: TrendDirection      # 周线趋势
    monthly_trend: TrendDirection     # 月线趋势
    is_valid: bool                    # 是否通过验证
    reason: str                       # 验证说明
    confidence_boost: float           # 置信度加成 (0-0.3)


@dataclass
class PreciseEntryZone:
    """精确入场区间"""
    level: str                        # 级别 (30min/5min)
    entry_low: float                  # 入场区间下沿
    entry_high: float                 # 入场区间上沿
    best_price: float                 # 最佳入场价
    stop_loss: float                  # 止损价
    reason: str                       # 理由
    buy_points: List[Dict]            # 该级别的买点列表


@dataclass
class ZoneTaoSignal:
    """区间套信号"""
    # 基础信息
    code: str
    name: str
    market: int
    
    # 日线买点（核心）
    daily_buy_point: DailyBuyPoint
    
    # 大级别验证
    trend_validation: TrendValidation
    
    # 精确入场区间
    entry_zones: List[PreciseEntryZone]
    
    # 综合评估
    total_confidence: float           # 综合置信度
    recommended_entry: float          # 推荐入场价
    stop_loss: float                  # 推荐止损
    target_price: float               # 目标价
    expected_profit: float            # 预期收益率 %
    risk_reward: float                # 盈亏比
    
    # 操作建议
    position_size: str                # 建议仓位 (轻仓/标准/重仓)
    urgency: str                      # 紧迫程度 (高/中/低)
    
    analysis_time: str


class ZoneTaoAnalyzer:
    """区间套分析器 v2.0"""
    
    def __init__(self, use_mysql: bool = True):
        """初始化"""
        self.analyzer = StockAnalyzer(use_mysql=use_mysql)
        logger.info("区间套分析器初始化完成")
    
    def analyze_stock(
        self,
        code: str,
        name: str,
        market: int,
        fetch_realtime: bool = True
    ) -> Optional[ZoneTaoSignal]:
        """
        完整的区间套分析
        
        Args:
            code: 股票代码
            name: 股票名称  
            market: 市场代码
            fetch_realtime: 是否实时获取5分钟数据
        
        Returns:
            ZoneTaoSignal 或 None
        """
        logger.info(f"开始区间套分析: {code} {name}")
        
        # ===== 第1步：日线找买点 =====
        daily_buy_points = self._find_daily_buy_points(code, name, market)
        
        if not daily_buy_points:
            logger.info(f"{code} 日线无买点，跳过")
            return None
        
        # 选择最强的买点
        best_daily_point = max(daily_buy_points, key=lambda x: x.confidence)
        logger.info(
            f"✅ {code} 日线发现{best_daily_point.buy_type.value}: "
            f"价格={best_daily_point.price:.2f}, "
            f"时间={best_daily_point.time}"
        )
        
        # ===== 第2步：大级别验证趋势 =====
        trend_validation = self._validate_large_level_trend(code, market)
        
        if not trend_validation.is_valid:
            logger.info(f"{code} 大级别趋势验证失败: {trend_validation.reason}")
            return None
        
        logger.info(
            f"✅ {code} 趋势验证通过: 周线={trend_validation.weekly_trend.value}, "
            f"月线={trend_validation.monthly_trend.value}"
        )
        
        # ===== 第3步：小级别找精确入场区间 =====
        entry_zones = []
        
        # 30分钟级别
        zone_30min = self._find_precise_entry_zone(
            code, name, market, '30min'
        )
        if zone_30min:
            entry_zones.append(zone_30min)
        
        # 5分钟级别（实时获取）
        if fetch_realtime:
            zone_5min = self._find_precise_entry_zone(
                code, name, market, '5min', realtime=True
            )
            if zone_5min:
                entry_zones.append(zone_5min)
        
        if not entry_zones:
            logger.warning(f"{code} 未找到精确入场区间")
            # 使用日线买点作为入场区间
            entry_zones.append(self._create_fallback_entry_zone(best_daily_point))
        
        # ===== 第4步：综合评估 =====
        signal = self._build_final_signal(
            code, name, market,
            best_daily_point,
            trend_validation,
            entry_zones
        )
        
        logger.info(
            f"🎯 {code} 区间套信号生成完成！"
            f"综合置信度={signal.total_confidence:.1%}, "
            f"推荐入场={signal.recommended_entry:.2f}, "
            f"预期收益={signal.expected_profit:.1f}%"
        )
        
        return signal
    
    def _find_daily_buy_points(
        self,
        code: str,
        name: str,
        market: int
    ) -> List[DailyBuyPoint]:
        """在日线找买点"""
        try:
            result = self.analyzer.analyze_stock(
                stock_code=code,
                stock_name=name,
                market=market,
                period='daily'
            )
            
            if not result or 'signals' not in result:
                return []
            
            signals = result['signals']
            last_price = result['last_price']
            
            buy_points = []
            
            # 提取各类买点
            for buy_type_key, buy_type_enum in [
                ('first_type', BuyPointType.FIRST),
                ('second_type', BuyPointType.SECOND),
                ('third_type', BuyPointType.THIRD)
            ]:
                if buy_type_key not in signals['buy']:
                    continue
                
                buy_signals = signals['buy'][buy_type_key]
                if not buy_signals:
                    continue
                
                # 只取最近的买点
                latest_signal = max(
                    buy_signals,
                    key=lambda x: pd.to_datetime(x.get('time', '1970-01-01'))
                )
                
                # 计算置信度
                base_confidence = {
                    BuyPointType.FIRST: 0.6,
                    BuyPointType.SECOND: 0.8,
                    BuyPointType.THIRD: 0.7,
                }
                
                buy_point = DailyBuyPoint(
                    code=code,
                    name=name,
                    buy_type=buy_type_enum,
                    price=float(latest_signal.get('price', last_price)),
                    time=str(latest_signal.get('time', '')).split()[0],
                    reason=latest_signal.get('reason', f'{buy_type_enum.value}信号'),
                    confidence=base_confidence[buy_type_enum]
                )
                
                buy_points.append(buy_point)
            
            return buy_points
        
        except Exception as e:
            logger.error(f"日线买点识别失败 {code}: {e}")
            return []
    
    def _validate_large_level_trend(
        self,
        code: str,
        market: int
    ) -> TrendValidation:
        """
        验证大级别趋势（周线）
        注意：如果数据库没有周线/月线数据，可以从日线数据推导
        """
        try:
            # 尝试获取周线数据
            weekly_result = self.analyzer.analyze_stock(
                stock_code=code,
                stock_name='',
                market=market,
                period='weekly'  # 如果不支持，会失败
            )
            
            if weekly_result:
                weekly_trend = self._judge_trend(weekly_result)
            else:
                # 从日线数据推导周线趋势
                daily_result = self.analyzer.analyze_stock(
                    stock_code=code,
                    stock_name='',
                    market=market,
                    period='daily'
                )
                weekly_trend = self._judge_trend_from_daily(daily_result, period=5)
            
            # 月线趋势（暂时用周线代替，或从日线推导）
            monthly_trend = self._judge_trend_from_daily(
                self.analyzer.analyze_stock(code, '', market, 'daily'),
                period=20
            )
            
            # 判断是否在上涨段
            is_valid = (
                weekly_trend == TrendDirection.UP or
                (weekly_trend == TrendDirection.SIDEWAYS and 
                 monthly_trend == TrendDirection.UP)
            )
            
            if is_valid:
                reason = f"周线{weekly_trend.value}，月线{monthly_trend.value}，符合上涨条件"
                confidence_boost = 0.2 if weekly_trend == TrendDirection.UP else 0.1
            else:
                reason = f"周线{weekly_trend.value}，月线{monthly_trend.value}，不在上涨段"
                confidence_boost = 0
            
            return TrendValidation(
                weekly_trend=weekly_trend,
                monthly_trend=monthly_trend,
                is_valid=is_valid,
                reason=reason,
                confidence_boost=confidence_boost
            )
        
        except Exception as e:
            logger.warning(f"趋势验证失败 {code}: {e}，默认通过")
            # 如果验证失败，默认通过但不加分
            return TrendValidation(
                weekly_trend=TrendDirection.UNKNOWN,
                monthly_trend=TrendDirection.UNKNOWN,
                is_valid=True,
                reason="无法获取周线数据，默认通过",
                confidence_boost=0
            )
    
    def _judge_trend(self, result: Dict) -> TrendDirection:
        """判断趋势方向"""
        if not result or 'signals' not in result:
            return TrendDirection.UNKNOWN
        
        signals = result['signals']
        
        # 统计最近的买卖点
        recent_buys = 0
        recent_sells = 0
        
        for buy_type in ['first_type', 'second_type', 'third_type']:
            if buy_type in signals['buy']:
                recent_buys += len(signals['buy'][buy_type])
        
        for sell_type in ['first_type', 'second_type', 'third_type']:
            if sell_type in signals['sell']:
                recent_sells += len(signals['sell'][sell_type])
        
        # 简单判断
        if recent_buys > recent_sells * 1.5:
            return TrendDirection.UP
        elif recent_sells > recent_buys * 1.5:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def _judge_trend_from_daily(
        self,
        result: Dict,
        period: int = 5
    ) -> TrendDirection:
        """从日线数据推导周线/月线趋势"""
        # 简化版：看均线方向
        # 你可以根据实际需求优化这个逻辑
        return self._judge_trend(result)
    
    def _find_precise_entry_zone(
        self,
        code: str,
        name: str,
        market: int,
        level: str,
        realtime: bool = False
    ) -> Optional[PreciseEntryZone]:
        """
        寻找精确入场区间
        
        Args:
            level: '30min' 或 '5min'
            realtime: 是否实时获取（用于5分钟）
        """
        try:
            result = self.analyzer.analyze_stock(
                stock_code=code,
                stock_name=name,
                market=market,
                period=level
            )
            
            if not result or 'signals' not in result:
                return None
            
            signals = result['signals']
            last_price = result['last_price']
            
            # 收集所有买点
            all_buy_points = []
            
            for buy_type_key in ['first_type', 'second_type', 'third_type']:
                if buy_type_key in signals['buy']:
                    for signal in signals['buy'][buy_type_key]:
                        all_buy_points.append({
                            'time': signal.get('time'),
                            'price': float(signal.get('price', last_price)),
                            'type': buy_type_key,
                            'reason': signal.get('reason', '')
                        })
            
            if not all_buy_points:
                return None
            
            # 计算入场区间（取最近买点附近的价格区间）
            recent_prices = [p['price'] for p in all_buy_points[-3:]]  # 最近3个买点
            
            entry_low = min(recent_prices) * 0.98   # 下沿：最低价-2%
            entry_high = max(recent_prices) * 1.02  # 上沿：最高价+2%
            best_price = sum(recent_prices) / len(recent_prices)  # 平均价
            
            # 止损位：区间下沿再-5%
            stop_loss = entry_low * 0.95
            
            return PreciseEntryZone(
                level=level,
                entry_low=entry_low,
                entry_high=entry_high,
                best_price=best_price,
                stop_loss=stop_loss,
                reason=f"{level}级别发现{len(all_buy_points)}个买点",
                buy_points=all_buy_points
            )
        
        except Exception as e:
            logger.error(f"寻找{level}入场区间失败 {code}: {e}")
            return None
    
    def _create_fallback_entry_zone(
        self,
        daily_point: DailyBuyPoint
    ) -> PreciseEntryZone:
        """创建兜底的入场区间（基于日线买点）"""
        price = daily_point.price
        
        return PreciseEntryZone(
            level='daily',
            entry_low=price * 0.97,
            entry_high=price * 1.03,
            best_price=price,
            stop_loss=price * 0.92,
            reason="使用日线买点作为入场区间",
            buy_points=[{
                'time': daily_point.time,
                'price': price,
                'type': daily_point.buy_type.value,
                'reason': daily_point.reason
            }]
        )
    
    def _build_final_signal(
        self,
        code: str,
        name: str,
        market: int,
        daily_buy_point: DailyBuyPoint,
        trend_validation: TrendValidation,
        entry_zones: List[PreciseEntryZone]
    ) -> ZoneTaoSignal:
        """构建最终信号"""
        
        # 综合置信度
        base_confidence = daily_buy_point.confidence
        trend_boost = trend_validation.confidence_boost
        zone_boost = 0.1 if len(entry_zones) >= 2 else 0  # 多级别共振加分
        
        total_confidence = min(base_confidence + trend_boost + zone_boost, 0.95)
        
        # 推荐入场价（选择最小级别的最佳价格）
        if entry_zones:
            # 优先用5分钟，其次30分钟
            best_zone = entry_zones[-1]  # 最后一个是最小级别
            recommended_entry = best_zone.best_price
            stop_loss = best_zone.stop_loss
        else:
            recommended_entry = daily_buy_point.price
            stop_loss = daily_buy_point.price * 0.92
        
        # 目标价（根据买点类型）
        if daily_buy_point.buy_type == BuyPointType.FIRST:
            target_multiplier = 1.15  # 一买目标15%
        elif daily_buy_point.buy_type == BuyPointType.SECOND:
            target_multiplier = 1.12  # 二买目标12%
        else:  # THIRD
            target_multiplier = 1.08  # 三买目标8%
        
        target_price = recommended_entry * target_multiplier
        
        # 预期收益和盈亏比
        expected_profit = ((target_price - recommended_entry) / recommended_entry) * 100
        risk = abs((stop_loss - recommended_entry) / recommended_entry) * 100
        risk_reward = expected_profit / risk if risk > 0 else 0
        
        # 仓位建议
        if total_confidence >= 0.8 and risk_reward >= 2.5:
            position_size = "重仓"
            urgency = "高"
        elif total_confidence >= 0.7 and risk_reward >= 2.0:
            position_size = "标准"
            urgency = "中"
        else:
            position_size = "轻仓"
            urgency = "低"
        
        return ZoneTaoSignal(
            code=code,
            name=name,
            market=market,
            daily_buy_point=daily_buy_point,
            trend_validation=trend_validation,
            entry_zones=entry_zones,
            total_confidence=total_confidence,
            recommended_entry=recommended_entry,
            stop_loss=stop_loss,
            target_price=target_price,
            expected_profit=expected_profit,
            risk_reward=risk_reward,
            position_size=position_size,
            urgency=urgency,
            analysis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def scan_market(
        self,
        stock_list: List[Tuple[str, str, int]],
        min_confidence: float = 0.7
    ) -> List[ZoneTaoSignal]:
        """
        全市场扫描
        
        Args:
            stock_list: [(code, name, market), ...]
            min_confidence: 最小置信度阈值
        
        Returns:
            符合条件的信号列表（按置信度降序）
        """
        logger.info(f"开始全市场区间套扫描，共{len(stock_list)}只股票")
        
        signals = []
        
        for code, name, market in stock_list:
            try:
                signal = self.analyze_stock(code, name, market)
                
                if signal and signal.total_confidence >= min_confidence:
                    signals.append(signal)
                    logger.info(
                        f"✅ {code} {name} - "
                        f"置信度:{signal.total_confidence:.1%} "
                        f"入场:{signal.recommended_entry:.2f} "
                        f"预期:{signal.expected_profit:.1f}%"
                    )
            
            except Exception as e:
                logger.error(f"分析 {code} 时出错: {e}")
                continue
        
        # 按置信度降序排序
        signals.sort(key=lambda x: x.total_confidence, reverse=True)
        
        logger.info(f"扫描完成，发现{len(signals)}个高质量信号")
        
        return signals
    
    def close(self):
        """关闭"""
        pass


def format_zonetao_report(signal: ZoneTaoSignal) -> str:
    """格式化区间套报告"""
    report = f"""
{'='*70}
【区间套信号】{signal.code} {signal.name}
{'='*70}

📊 日线买点（核心信号）:
  • 类型: {signal.daily_buy_point.buy_type.value}
  • 价格: ¥{signal.daily_buy_point.price:.2f}
  • 时间: {signal.daily_buy_point.time}
  • 理由: {signal.daily_buy_point.reason}
  • 基础置信度: {signal.daily_buy_point.confidence:.1%}

🔍 大级别趋势验证:
  • 周线趋势: {signal.trend_validation.weekly_trend.value}
  • 月线趋势: {signal.trend_validation.monthly_trend.value}
  • 验证结果: {'✅ 通过' if signal.trend_validation.is_valid else '❌ 未通过'}
  • 说明: {signal.trend_validation.reason}
  • 置信度加成: +{signal.trend_validation.confidence_boost:.1%}

🎯 精确入场区间:
"""
    
    for i, zone in enumerate(signal.entry_zones, 1):
        report += f"""
  区间 {i} - [{zone.level}级别]
    • 入场区间: ¥{zone.entry_low:.2f} ~ ¥{zone.entry_high:.2f}
    • 最佳价格: ¥{zone.best_price:.2f}
    • 止损位: ¥{zone.stop_loss:.2f}
    • 理由: {zone.reason}
    • 买点数: {len(zone.buy_points)}个
"""
    
    report += f"""
💰 操作建议:
  • 综合置信度: {signal.total_confidence:.1%}
  • 推荐入场价: ¥{signal.recommended_entry:.2f}
  • 建议止损: ¥{signal.stop_loss:.2f}
  • 目标价: ¥{signal.target_price:.2f}
  • 预期收益: +{signal.expected_profit:.1f}%
  • 盈亏比: {signal.risk_reward:.2f}:1
  
  • 建议仓位: {signal.position_size}
  • 紧迫程度: {signal.urgency}

⏰ 分析时间: {signal.analysis_time}

{'='*70}
"""
    
    return report


if __name__ == '__main__':
    # 测试代码
    analyzer = ZoneTaoAnalyzer(use_mysql=True)
    
    signal = analyzer.analyze_stock(
        code='600000',
        name='浦发银行',
        market=1,
        fetch_realtime=True
    )
    
    if signal:
        print(format_zonetao_report(signal))
    else:
        print("未发现符合条件的信号")
    
    analyzer.close()
