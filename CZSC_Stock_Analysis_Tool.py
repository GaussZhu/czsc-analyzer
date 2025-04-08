# -*- coding: utf-8 -*-
"""
CZSC Stock Analysis Tool
Based on Cheng Zhong Shuo Chan Technical Analysis Theory

This script monitors Beijing Stock Exchange stocks for buy/sell signals
on daily and 30-minute timeframes using CZSC theory.
"""

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
from pytdx.util.best_ip import select_best_ip
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class CZSCAnalysis:
    def __init__(self):
        """Initialize CZSC Analysis with TDX API connection"""
        self.api = TdxHq_API()
        self.connected = False
        self.best_ip = None
        # 初始化服务器列表（动态更新）
        self.servers = [
            ('sztdx.gtjas.com', 7709),
            ('60.12.136.250', 7709),
            ('60.191.117.167', 7709),
            ('218.75.126.9', 7709),
            ('115.238.56.198', 7709),
            ('115.238.90.165', 7709)
        ]
        # List of Beijing Stock Exchange stocks to monitor
        self.stock_list = [
            {'code': '600000', 'name': '浦发银行', 'market': TDXParams.MARKET_SH},
            {'code': '600004', 'name': '白云机场', 'market': TDXParams.MARKET_SH},
            {'code': '600006', 'name': '东风股份', 'market': TDXParams.MARKET_SH},
            # Add more stocks as needed
        ]
    def update_servers(self):
            """动态更新服务器列表，将最优服务器置顶"""
            print("⌛ 正在检测最优服务器...")
            best = select_best_ip()
            if best:
                self.best_ip = (best['ip'], best['port'])
                # 如果最优服务器不在列表中则插入首位
                if self.best_ip not in self.servers:
                    self.servers.insert(0, self.best_ip)
                print(f"✅ 最优服务器: {best['ip']}:{best['port']}")
            else:
                print("⚠️ 使用默认服务器列表")

    def connect(self, retry=3):
        """智能连接方法（整合最优服务器检测）"""
        if self.connected:
            return True
        
        self.update_servers()  # 每次连接前更新服务器列表

        for attempt in range(retry):
            print(f"\n→ 第 {attempt+1}/{retry} 次连接尝试")
            
            # 优先尝试最优服务器
            for idx, (ip, port) in enumerate(self.servers, 1):
                try:
                    print(f"  尝试 {ip}:{port}...", end=' ')
                    self.connected = self.api.connect(ip, port, time_out=3)
                    if self.connected:
                        print("✓")
                        print(f"当前使用服务器: {ip}:{port}")
                        return True
                    print("×")
                except Exception as e:
                    print(f"连接异常: {str(e)}")
            
            if attempt < retry-1:
                print("等待重试...")
                time.sleep(5)
        
        print("所有服务器连接失败，请检查网络")
        return False

    def is_trading_day(self):
        """智能判断交易日（简单版，未处理节假日）"""
        now = datetime.datetime.now()
        if now.weekday() >= 5:  # 周六日
            print(f"{now.date()} 非交易日（周末）")
            return False
        # TODO: 此处可添加节假日判断逻辑
        return True

    def in_trading_hours(self):
        """判断是否在交易时段内"""
        now = datetime.datetime.now()
        morning_start = datetime.time(9, 25)
        morning_end = datetime.time(11, 30)
        afternoon_start = datetime.time(12, 55)
        afternoon_end = datetime.time(15, 5)
        
        # 提前15分钟开始监控
        return (
            (morning_start <= now.time() <= morning_end) or
            (afternoon_start <= now.time() <= afternoon_end)
        )

    def wait_until_next_trading(self):
        """等待到下一个交易时段"""
        while True:
            now = datetime.datetime.now()
            next_morning = datetime.datetime.combine(
                now.date(), datetime.time(9,25))
            next_afternoon = datetime.datetime.combine(
                now.date(), datetime.time(12,55))
            
            # 如果当前时间早于上午时段
            if now < next_morning:
                wait_seconds = (next_morning - now).total_seconds()
            # 上午收盘后等待下午开盘
            elif now.time() < datetime.time(12,55):
                wait_seconds = (next_afternoon - now).total_seconds()
            # 否则等待次日
            else:
                wait_seconds = ((next_morning + datetime.timedelta(days=1)) - now).total_seconds()
            
            print(f"⏳ 下次检测时间：{datetime.datetime.now() + datetime.timedelta(seconds=wait_seconds)}")
            time.sleep(wait_seconds)
            
            # 等待结束后检查是否为交易日
            if self.is_trading_day():
                return

    def monitor_stocks(self, interval=300):
        """智能监控（仅在交易时段运行）"""
        while True:
            if not self.is_trading_day():
                print("💤 当前为非交易日，进入休眠")
                self.wait_until_next_trading()
                continue
                
            if not self.in_trading_hours():
                print("💤 当前为非交易时段")
                self.wait_until_next_trading()
                continue
                
            try:
                # 实际监控逻辑
                print(f"\n📈 开始监控 [{datetime.datetime.now()}]")
                # ...（保留原有监控逻辑）
                
            except Exception as e:
                print(f"监控异常: {str(e)}")
                time.sleep(60)
                
            print(f"⏳ 下次更新 {interval}秒后...")
            time.sleep(interval)        
    def connect(self, retry=3):
        """连接到TDX服务器（支持多服务器重试和超时控制）"""
        if self.connected:
            print("✓ 已经连接到TDX服务器")
            return True

        # 备选服务器列表（优先级从高到低）
        servers = [
            ('sztdx.gtjas.com', 7709),  # 默认主服务器
            ('60.12.136.250', 7709),  # 备用服务器1
            ('60.191.117.167', 7709),   # 备用服务器2
            ('218.75.126.9', 7709),    # 浙江电信
            ('115.238.56.198', 7709),   # 广州电信
            ('115.238.90.165', 7709)     # 湖南电信
        ]

        for attempt in range(retry):
            print(f"→ 尝试第 {attempt + 1} 次连接（共 {retry} 次）")
            
            for idx, (ip, port) in enumerate(servers, 1):
                try:
                    print(f"  正在尝试服务器 {idx}/{len(servers)}: {ip}:{port}...", end=' ')
                    
                    # 设置超时时间为3秒
                    self.connected = self.api.connect(ip, port, time_out=3)
                    
                    if self.connected:
                        print("✓ 连接成功")
                        print(f"★ 当前使用的服务器: {ip}:{port}")
                        return True
                    else:
                        print("× 连接失败（请检查网络或服务器状态）")
                        
                except Exception as e:
                    print(f"⚠ 连接异常: {str(e)}")
                    self.connected = False
                    
            if attempt < retry - 1:
                print(f"⏳ 等待2秒后重试...")
                time.sleep(2)

        print("✗ 所有服务器连接均失败，建议：")
        print("  1. 检查通达信客户端是否已运行")
        print("  2. 检查防火墙设置")
        print("  3. 稍后再试")
        return False
    
    def disconnect(self):
        """Disconnect from TDX server"""
        if self.connected:
            self.api.disconnect()
            self.connected = False
            print("Disconnected from TDX server")
    
    def get_stock_data(self, stock_info, period='daily', count=300):
        """获取股票数据（修复日期解析问题）"""
        if not self.connect():
            return None
        
        try:
            if period == 'daily':
                data_type = 9  # 日线数据
            elif period == '30min':
                data_type = 0  # 30分钟数据
            else:
                print(f"不支持的周期: {period}")
                return None

            data = self.api.get_security_bars(
                data_type, 
                stock_info['market'], 
                stock_info['code'], 
                0, 
                count
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 处理空数据
            if df.empty:
                print(f"获取到空数据: {stock_info['code']}")
                return None

            # 统一处理列名
            df.rename(columns={
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'vol': 'Volume',
                'amount': 'Amount'
            }, inplace=True)
            
            # 改进的日期处理逻辑
            def parse_tdx_datetime(x):
                try:
                    # 处理不同格式的日期字符串
                    if isinstance(x, int):
                        s = str(x).zfill(12)
                        return datetime.datetime.strptime(s, "%Y%m%d%H%M")
                    elif '-' in str(x):
                        return pd.to_datetime(x, format='mixed')
                    else:
                        return pd.to_datetime(str(x), format='%Y%m%d%H%M', errors='coerce')
                except:
                    return pd.NaT

            # 生成日期列
            if period == 'daily':
                # 日线数据：使用 year/month/day 组合日期
                df['Date'] = pd.to_datetime(
                    df[['year', 'month', 'day']]
                    .astype(str)
                    .apply(lambda x: f"{x['year']}-{x['month']}-{x['day']}", axis=1)
                )
            elif period == '30min':
                # 30分钟数据：应用灵活解析
                df['Date'] = df['datetime'].apply(parse_tdx_datetime)
                # 清理无效日期
                df = df[df['Date'].notna()]
                
            # 设置日期索引
            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']]
        
        except Exception as e:
            print(f"获取 {stock_info['code']} 数据失败: {str(e)}")
            return None
        
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        Calculate MACD indicator
        
        Args:
            data: DataFrame with 'Close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            DataFrame with added MACD columns
        """
        df = data.copy()
        
        # Calculate EMAs
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        
        # Calculate Signal line
        df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        df['Histogram'] = df['MACD'] - df['Signal']
        
        return df
    
    def identify_fractal_tops_bottoms(self, data, n=2):
        """识别分型（修复链式赋值警告）"""
        df = data.copy()
        
        # 使用loc初始化列避免链式操作
        df.loc[:, 'FractalTop'] = False
        df.loc[:, 'FractalBottom'] = False

        # 将索引转换为整数位置
        indexer = df.index.values
        
        # 识别顶分型
        for i in range(n, len(df) - n):
            current_high = df['High'].iloc[i]
            # 检查左右n根K线
            left_highs = df['High'].iloc[i-n:i]
            right_highs = df['High'].iloc[i+1:i+n+1]
            
            if all(current_high > left_highs) and all(current_high > right_highs):
                df.loc[indexer[i], 'FractalTop'] = True

        # 识别底分型
        for i in range(n, len(df) - n):
            current_low = df['Low'].iloc[i]
            left_lows = df['Low'].iloc[i-n:i]
            right_lows = df['Low'].iloc[i+1:i+n+1]
            
            if all(current_low < left_lows) and all(current_low < right_lows):
                df.loc[indexer[i], 'FractalBottom'] = True

        return df
    
    def identify_stroke(self, data):
        """
        Identify strokes (笔) from fractal tops and bottoms
        
        Args:
            data: DataFrame with fractal tops and bottoms marked
            
        Returns:
            List of stroke points
        """
        df = data.copy()
        strokes = []
        
        # Find all tops and bottoms
        tops = df[df['FractalTop']].index.tolist()
        bottoms = df[df['FractalBottom']].index.tolist()
        
        # Combine and sort chronologically
        points = [(idx, 'top', df.loc[idx, 'High']) for idx in tops] + \
                [(idx, 'bottom', df.loc[idx, 'Low']) for idx in bottoms]
        points.sort(key=lambda x: x[0])
        
        # Process points to form valid strokes
        if len(points) < 2:
            return strokes
        
        current = points[0]
        for i in range(1, len(points)):
            if points[i][1] != current[1]:  # Different type
                # Create a stroke
                strokes.append((current[0], points[i][0], current[1], points[i][1], 
                                current[2], points[i][2]))
                current = points[i]
        
        return strokes
    
    def identify_pivot_points(self, strokes):
        """
        Identify pivot points (中枢) from strokes
        
        Args:
            strokes: List of strokes
            
        Returns:
            List of pivot points
        """
        if len(strokes) < 3:
            return []
        
        pivot_points = []
        
        # Need at least 3 strokes to form a pivot point
        for i in range(len(strokes) - 2):
            # Check for overlapping in price range
            high1, low1 = max(strokes[i][4], strokes[i][5]), min(strokes[i][4], strokes[i][5])
            high2, low2 = max(strokes[i+1][4], strokes[i+1][5]), min(strokes[i+1][4], strokes[i+1][5])
            high3, low3 = max(strokes[i+2][4], strokes[i+2][5]), min(strokes[i+2][4], strokes[i+2][5])
            
            # Calculate overlap
            overlap_high = min(high1, high2, high3)
            overlap_low = max(low1, low2, low3)
            
            # Check if there's an overlap
            if overlap_high > overlap_low:
                # Create a pivot point
                start_time = strokes[i][0]
                end_time = strokes[i+2][1]
                pivot_points.append({
                    'start': start_time,
                    'end': end_time,
                    'high': overlap_high,
                    'low': overlap_low,
                    'strokes': [strokes[i], strokes[i+1], strokes[i+2]]
                })
        
        return pivot_points
    
    def detect_divergence(self, data, strokes):
        """
        Detect divergence (背驰) patterns
        
        Args:
            data: DataFrame with MACD data
            strokes: List of strokes
            
        Returns:
            List of divergence points
        """
        if len(strokes) < 4:
            return []
        
        divergences = []
        
        # Group strokes by direction (up or down)
        up_strokes = [s for s in strokes if s[2] == 'bottom' and s[3] == 'top']
        down_strokes = [s for s in strokes if s[2] == 'top' and s[3] == 'bottom']
        
        # Check for bullish divergence (price makes lower low but MACD makes higher low)
        for i in range(1, len(down_strokes)):
            current = down_strokes[i]
            previous = down_strokes[i-1]
            
            # Check if price made a lower low
            if current[5] < previous[5]:
                # Get MACD values at the end of each stroke
                current_macd = data.loc[current[1], 'Histogram']
                previous_macd = data.loc[previous[1], 'Histogram']
                
                # Check if MACD made a higher low (less negative)
                if current_macd > previous_macd and current_macd < 0:
                    divergences.append({
                        'type': 'bullish',
                        'time': current[1],
                        'price': current[5],
                        'previous_time': previous[1],
                        'previous_price': previous[5],
                        'macd': current_macd,
                        'previous_macd': previous_macd
                    })
        
        # Check for bearish divergence (price makes higher high but MACD makes lower high)
        for i in range(1, len(up_strokes)):
            current = up_strokes[i]
            previous = up_strokes[i-1]
            
            # Check if price made a higher high
            if current[5] > previous[5]:
                # Get MACD values at the end of each stroke
                current_macd = data.loc[current[1], 'Histogram']
                previous_macd = data.loc[previous[1], 'Histogram']
                
                # Check if MACD made a lower high
                if current_macd < previous_macd and current_macd > 0:
                    divergences.append({
                        'type': 'bearish',
                        'time': current[1],
                        'price': current[5],
                        'previous_time': previous[1],
                        'previous_price': previous[5],
                        'macd': current_macd,
                        'previous_macd': previous_macd
                    })
        
        return divergences
    
    def identify_buy_sell_points(self, data, pivot_points, divergences):
        """
        Identify buy and sell points based on CZSC theory
        
        Args:
            data: DataFrame with price data
            pivot_points: List of pivot points
            divergences: List of divergence points
            
        Returns:
            Dictionary with buy and sell points
        """
        buy_points = {
            'first_type': [],  # 第一类买点
            'second_type': [], # 第二类买点
            'third_type': []   # 第三类买点
        }
        
        sell_points = {
            'first_type': [],  # 第一类卖点
            'second_type': [], # 第二类卖点
            'third_type': []   # 第三类卖点
        }
        
        # First type buy points: Trend reversals with divergence
        for div in divergences:
            if div['type'] == 'bullish':
                buy_points['first_type'].append({
                    'time': div['time'],
                    'price': div['price'],
                    'reason': 'Bullish divergence at the end of a downtrend'
                })
        
        # First type sell points: Trend reversals with divergence
        for div in divergences:
            if div['type'] == 'bearish':
                sell_points['first_type'].append({
                    'time': div['time'],
                    'price': div['price'],
                    'reason': 'Bearish divergence at the end of an uptrend'
                })
        
        # Second type buy points: Retracement after a First type buy point
        for i, buy_point in enumerate(buy_points['first_type']):
            # Look for a pullback that doesn't go below the first type buy point
            idx = data.index.get_loc(buy_point['time'])
            if idx + 5 >= len(data):  # Need at least 5 bars after the first type point
                continue
                
            # Find the lowest point in the next 5-20 bars
            min_price = float('inf')
            min_idx = None
            for j in range(idx + 5, min(idx + 20, len(data))):
                if data['Low'].iloc[j] < min_price:
                    min_price = data['Low'].iloc[j]
                    min_idx = j
            
            if min_idx and min_price > buy_point['price']:
                buy_points['second_type'].append({
                    'time': data.index[min_idx],
                    'price': min_price,
                    'reason': 'Retracement after first type buy point'
                })
        
        # Third type buy points: Failed test of pivot point low
        for pivot in pivot_points:
            idx = data.index.get_loc(pivot['end'])
            if idx + 5 >= len(data):
                continue
                
            # Look for a test of the pivot low that fails (doesn't go below)
            for j in range(idx + 5, min(idx + 30, len(data))):
                if data['Low'].iloc[j] < pivot['low'] * 1.05 and data['Low'].iloc[j] > pivot['low']:
                    buy_points['third_type'].append({
                        'time': data.index[j],
                        'price': data['Low'].iloc[j],
                        'reason': 'Failed test of pivot point low'
                    })
                    break
        
        # Similarly for sell points (reverse logic)
        for pivot in pivot_points:
            idx = data.index.get_loc(pivot['end'])
            if idx + 5 >= len(data):
                continue
                
            # Look for a test of the pivot high that fails (doesn't go above)
            for j in range(idx + 5, min(idx + 30, len(data))):
                if data['High'].iloc[j] > pivot['high'] * 0.95 and data['High'].iloc[j] < pivot['high']:
                    sell_points['third_type'].append({
                        'time': data.index[j],
                        'price': data['High'].iloc[j],
                        'reason': 'Failed test of pivot point high'
                    })
                    break
        
        return {'buy': buy_points, 'sell': sell_points}
    
    def analyze_stock(self, stock_info, period='daily'):
        """
        Complete analysis of a stock
        
        Args:
            stock_info: Dictionary with stock code and market
            period: 'daily' or '30min'
            
        Returns:
            Analysis results
        """
        # Get data
        data = self.get_stock_data(stock_info, period)
        if data is None or len(data) < 30:
            return None
        
        # Calculate MACD
        data = self.calculate_macd(data)
        
        # Identify fractals
        data = self.identify_fractal_tops_bottoms(data)
        
        # Identify strokes
        strokes = self.identify_stroke(data)
        
        # Identify pivot points
        pivot_points = self.identify_pivot_points(strokes)
        
        # Detect divergence
        divergences = self.detect_divergence(data, strokes)
        
        # Identify buy/sell points
        signals = self.identify_buy_sell_points(data, pivot_points, divergences)
        
        # Prepare results
        result = {
            'stock': stock_info,
            'period': period,
            'last_price': data['Close'].iloc[-1] if len(data) > 0 else None,
            'pivot_points': pivot_points,
            'divergences': divergences,
            'signals': signals
        }
        # 在返回结果前添加时间过滤
        valid_start = datetime.datetime.now() - datetime.timedelta(days=3)
        result['signals'] = {
            'buy': {
                k: [p for p in v if p['time'] > valid_start] 
                for k, v in signals['buy'].items()
            },
            'sell': {
                k: [p for p in v if p['time'] > valid_start]
                for k, v in signals['sell'].items()
            },
            'start': valid_start,
            'end': datetime.datetime.now()
        }
        return result
    
    def print_analysis_results(self, results, max_display=3):
        """优化结果显示（控制输出数量）"""
        print(f"\n{'='*80}")
        print(f"股票: {results['stock']['name']} ({results['stock']['code']})")
        print(f"周期: {results['period']} | 最新价: {results['last_price']}")
        print(f"信号时间范围: {results['signals']['start']} 至 {results['signals']['end']}")
        print('-'*80)

        # 关键信号摘要
        print("\n关键信号摘要：")
        signal_counts = {
            '买点': sum(len(v) for v in results['signals']['buy'].values()),
            '卖点': sum(len(v) for v in results['signals']['sell'].values())
        }
        print(f"★ 发现 {signal_counts['买点']} 个买入信号 | {signal_counts['卖点']} 个卖出信号")

        # 精简显示逻辑
        def print_signals(signal_type, signals):
            if signals:
                print(f"\n{signal_type}信号（最近{max_display}个）：")
                for s in signals[:max_display]:
                    time_str = s['time'].strftime('%m-%d %H:%M') 
                    print(f"→ {time_str} | 价格: {s['price']:.2f} | 类型: {s['type']}")

        # 合并所有信号并按时间排序
        all_signals = []
        for stype, points in results['signals']['buy'].items():
            all_signals.extend([{'time':p['time'], 'price':p['price'], 'type':f'买点_{stype}'} for p in points])
        for stype, points in results['signals']['sell'].items():
            all_signals.extend([{'time':p['time'], 'price':p['price'], 'type':f'卖点_{stype}'} for p in points])
        
        # 按时间倒序排列
        all_signals.sort(key=lambda x: x['time'], reverse=True)
        
        # 显示最近信号
        print_signals("最新", all_signals[:max_display])

        # 显示重要提示
        if results['divergences']:
            print("\n重要提示：")
            for d in results['divergences']:
                print(f"❗ {d['time'].strftime('%m-%d %H:%M')} 发现{d['type']}背离")

        print(f"\n{'='*80}")
    def monitor_stocks(self, interval=300):
        """
        Monitor stocks for buy/sell signals
        
        Args:
            interval: Refresh interval in seconds
        """
        print(f"Starting monitoring at {datetime.datetime.now()}")
        print(f"Monitoring {len(self.stock_list)} stocks with refresh interval of {interval} seconds")
        
        try:
            while True:
                print(f"\n{'-'*40}")
                print(f"Refreshing data at {datetime.datetime.now()}")
                
                for stock in self.stock_list:
                    print(f"\nAnalyzing {stock['name']} ({stock['code']})")
                    
                    # Daily analysis
                    daily_results = self.analyze_stock(stock, 'daily')
                    if daily_results:
                        # Check for recent signals (last 3 days)
                        recent_buy = False
                        recent_sell = False
                        
                        for type_name, points in daily_results['signals']['buy'].items():
                            for point in points:
                                if (datetime.datetime.now() - point['time']).days <= 3:
                                    recent_buy = True
                                    print(f"ALERT: Recent {type_name} buy signal on daily chart at {point['time']}")
                                    print(f"       Price: {point['price']:.2f}, Reason: {point['reason']}")
                        
                        for type_name, points in daily_results['signals']['sell'].items():
                            for point in points:
                                if (datetime.datetime.now() - point['time']).days <= 3:
                                    recent_sell = True
                                    print(f"ALERT: Recent {type_name} sell signal on daily chart at {point['time']}")
                                    print(f"       Price: {point['price']:.2f}, Reason: {point['reason']}")
                        
                        # Only print full results if there are recent signals
                        if recent_buy or recent_sell:
                            self.print_analysis_results(daily_results)
                    
                    # 30-minute analysis
                    min30_results = self.analyze_stock(stock, '30min')
                    if min30_results:
                        # Check for recent signals (last 6 hours)
                        recent_buy = False
                        recent_sell = False
                        
                        for type_name, points in min30_results['signals']['buy'].items():
                            for point in points:
                                if (datetime.datetime.now() - point['time']).total_seconds() <= 6*3600:
                                    recent_buy = True
                                    print(f"ALERT: Recent {type_name} buy signal on 30-min chart at {point['time']}")
                                    print(f"       Price: {point['price']:.2f}, Reason: {point['reason']}")
                        
                        for type_name, points in min30_results['signals']['sell'].items():
                            for point in points:
                                if (datetime.datetime.now() - point['time']).total_seconds() <= 6*3600:
                                    recent_sell = True
                                    print(f"ALERT: Recent {type_name} sell signal on 30-min chart at {point['time']}")
                                    print(f"       Price: {point['price']:.2f}, Reason: {point['reason']}")
                        
                        # Only print full results if there are recent signals
                        if recent_buy or recent_sell:
                            self.print_analysis_results(min30_results)
                
                print(f"\nSleeping for {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.disconnect()

if __name__ == "__main__":
    analyzer = CZSCAnalysis()
    
    # Option 1: Run continuous monitoring
    analyzer.monitor_stocks(interval=300)  # Check every 5 minutes
    
    # Option 2: Analyze a specific stock once
    # stock_info = {'code': '430047', 'name': '诺思兰德', 'market': TDXParams.MARKET_BJ}
    # daily_results = analyzer.analyze_stock(stock_info, 'daily')
    # analyzer.print_analysis_results(daily_results)
    # min30_results = analyzer.analyze_stock(stock_info, '30min')
    # analyzer.print_analysis_results(min30_results)
    
    # Disconnect when done
    analyzer.disconnect()
