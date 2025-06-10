"""Core analysis and monitoring logic for CZSC Stock Analysis Tool."""

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
    """Encapsulates all analysis functionality."""

    def __init__(self):
        self.api = TdxHq_API()
        self.connected = False
        self.best_ip = None
        self.servers = [
            ('sztdx.gtjas.com', 7709),
            ('60.12.136.250', 7709),
            ('60.191.117.167', 7709),
            ('218.75.126.9', 7709),
            ('115.238.56.198', 7709),
            ('115.238.90.165', 7709),
        ]
        self.stock_list = [
            {'code': '600000', 'name': '浦发银行', 'market': TDXParams.MARKET_SH},
            {'code': '600004', 'name': '白云机场', 'market': TDXParams.MARKET_SH},
            {'code': '600006', 'name': '东风股份', 'market': TDXParams.MARKET_SH},
        ]

    # ------------------------------------------------------------------
    # Connection and data utilities
    # ------------------------------------------------------------------
    def update_servers(self):
        """Check best TDX server and prepend it to the list."""
        print("⌛ 正在检测最优服务器...")
        best = select_best_ip()
        if best:
            self.best_ip = (best['ip'], best['port'])
            if self.best_ip not in self.servers:
                self.servers.insert(0, self.best_ip)
            print(f"✅ 最优服务器: {best['ip']}:{best['port']}")
        else:
            print("⚠️ 使用默认服务器列表")

    def connect(self, retry: int = 3) -> bool:
        """Connect to TDX server with retries."""
        if self.connected:
            return True

        self.update_servers()
        for attempt in range(retry):
            print(f"\n→ 第 {attempt + 1}/{retry} 次连接尝试")
            for idx, (ip, port) in enumerate(self.servers, 1):
                try:
                    print(f"  尝试 {ip}:{port}...", end=" ")
                    self.connected = self.api.connect(ip, port, time_out=3)
                    if self.connected:
                        print("✓")
                        print(f"当前使用服务器: {ip}:{port}")
                        return True
                    print("×")
                except Exception as e:
                    print(f"连接异常: {str(e)}")
            if attempt < retry - 1:
                print("等待重试...")
                time.sleep(5)
        print("所有服务器连接失败，请检查网络")
        return False

    def disconnect(self):
        if self.connected:
            self.api.disconnect()
            self.connected = False
            print("Disconnected from TDX server")

    def get_stock_data(self, stock_info, period: str = 'daily', count: int = 300):
        """Fetch stock data from TDX."""
        if not self.connect():
            return None

        try:
            if period == 'daily':
                data_type = 9
            elif period == '30min':
                data_type = 0
            else:
                print(f"不支持的周期: {period}")
                return None

            data = self.api.get_security_bars(
                data_type,
                stock_info['market'],
                stock_info['code'],
                0,
                count,
            )
            df = pd.DataFrame(data)
            if df.empty:
                print(f"获取到空数据: {stock_info['code']}")
                return None
            df.rename(columns={
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'vol': 'Volume',
                'amount': 'Amount',
            }, inplace=True)

            def parse_tdx_datetime(x):
                try:
                    if isinstance(x, int):
                        s = str(x).zfill(12)
                        return datetime.datetime.strptime(s, "%Y%m%d%H%M")
                    elif '-' in str(x):
                        return pd.to_datetime(x, format='mixed')
                    else:
                        return pd.to_datetime(str(x), format='%Y%m%d%H%M', errors='coerce')
                except Exception:
                    return pd.NaT

            if period == 'daily':
                df['Date'] = pd.to_datetime(
                    df[['year', 'month', 'day']].astype(str).apply(
                        lambda x: f"{x['year']}-{x['month']}-{x['day']}", axis=1
                    )
                )
            else:
                df['Date'] = df['datetime'].apply(parse_tdx_datetime)
                df = df[df['Date'].notna()]

            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']]
        except Exception as e:
            print(f"获取 {stock_info['code']} 数据失败: {str(e)}")
            return None

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        df = data.copy()
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        return df

    @staticmethod
    def identify_fractal_tops_bottoms(data: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        df = data.copy()
        df.loc[:, 'FractalTop'] = False
        df.loc[:, 'FractalBottom'] = False
        indexer = df.index.values
        for i in range(n, len(df) - n):
            current_high = df['High'].iloc[i]
            left_highs = df['High'].iloc[i - n:i]
            right_highs = df['High'].iloc[i + 1:i + n + 1]
            if all(current_high > left_highs) and all(current_high > right_highs):
                df.loc[indexer[i], 'FractalTop'] = True
        for i in range(n, len(df) - n):
            current_low = df['Low'].iloc[i]
            left_lows = df['Low'].iloc[i - n:i]
            right_lows = df['Low'].iloc[i + 1:i + n + 1]
            if all(current_low < left_lows) and all(current_low < right_lows):
                df.loc[indexer[i], 'FractalBottom'] = True
        return df

    @staticmethod
    def identify_stroke(data: pd.DataFrame):
        df = data.copy()
        strokes = []
        tops = df[df['FractalTop']].index.tolist()
        bottoms = df[df['FractalBottom']].index.tolist()
        points = [(idx, 'top', df.loc[idx, 'High']) for idx in tops] + \
            [(idx, 'bottom', df.loc[idx, 'Low']) for idx in bottoms]
        points.sort(key=lambda x: x[0])
        if len(points) < 2:
            return strokes
        current = points[0]
        for i in range(1, len(points)):
            if points[i][1] != current[1]:
                strokes.append((current[0], points[i][0], current[1], points[i][1], current[2], points[i][2]))
                current = points[i]
        return strokes

    @staticmethod
    def identify_pivot_points(strokes):
        if len(strokes) < 3:
            return []
        pivot_points = []
        for i in range(len(strokes) - 2):
            high1, low1 = max(strokes[i][4], strokes[i][5]), min(strokes[i][4], strokes[i][5])
            high2, low2 = max(strokes[i + 1][4], strokes[i + 1][5]), min(strokes[i + 1][4], strokes[i + 1][5])
            high3, low3 = max(strokes[i + 2][4], strokes[i + 2][5]), min(strokes[i + 2][4], strokes[i + 2][5])
            overlap_high = min(high1, high2, high3)
            overlap_low = max(low1, low2, low3)
            if overlap_high > overlap_low:
                start_time = strokes[i][0]
                end_time = strokes[i + 2][1]
                pivot_points.append({
                    'start': start_time,
                    'end': end_time,
                    'high': overlap_high,
                    'low': overlap_low,
                    'strokes': [strokes[i], strokes[i + 1], strokes[i + 2]],
                })
        return pivot_points

    @staticmethod
    def detect_divergence(data: pd.DataFrame, strokes):
        if len(strokes) < 4:
            return []
        divergences = []
        up_strokes = [s for s in strokes if s[2] == 'bottom' and s[3] == 'top']
        down_strokes = [s for s in strokes if s[2] == 'top' and s[3] == 'bottom']
        for i in range(1, len(down_strokes)):
            current = down_strokes[i]
            previous = down_strokes[i - 1]
            if current[5] < previous[5]:
                current_macd = data.loc[current[1], 'Histogram']
                previous_macd = data.loc[previous[1], 'Histogram']
                if current_macd > previous_macd and current_macd < 0:
                    divergences.append({
                        'type': 'bullish',
                        'time': current[1],
                        'price': current[5],
                        'previous_time': previous[1],
                        'previous_price': previous[5],
                        'macd': current_macd,
                        'previous_macd': previous_macd,
                    })
        for i in range(1, len(up_strokes)):
            current = up_strokes[i]
            previous = up_strokes[i - 1]
            if current[4] > previous[4]:
                current_macd = data.loc[current[1], 'Histogram']
                previous_macd = data.loc[previous[1], 'Histogram']
                if current_macd < previous_macd and current_macd > 0:
                    divergences.append({
                        'type': 'bearish',
                        'time': current[1],
                        'price': current[5],
                        'previous_time': previous[1],
                        'previous_price': previous[5],
                        'macd': current_macd,
                        'previous_macd': previous_macd,
                    })
        return divergences

    @staticmethod
    def identify_buy_sell_points(data, pivot_points, divergences):
        buy_points = {"first_type": [], "second_type": [], "third_type": []}
        sell_points = {"first_type": [], "second_type": [], "third_type": []}
        for div in divergences:
            if div['type'] == 'bullish':
                buy_points['first_type'].append({
                    'time': div['time'],
                    'price': div['price'],
                    'reason': 'Bullish divergence at the end of a downtrend',
                })
            if div['type'] == 'bearish':
                sell_points['first_type'].append({
                    'time': div['time'],
                    'price': div['price'],
                    'reason': 'Bearish divergence at the end of an uptrend',
                })
        for i, buy_point in enumerate(buy_points['first_type']):
            idx = data.index.get_loc(buy_point['time'])
            if idx + 5 >= len(data):
                continue
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
                    'reason': 'Retracement after first type buy point',
                })
        for pivot in pivot_points:
            idx = data.index.get_loc(pivot['end'])
            if idx + 5 >= len(data):
                continue
            for j in range(idx + 5, min(idx + 30, len(data))):
                if data['Low'].iloc[j] < pivot['low'] * 1.05 and data['Low'].iloc[j] > pivot['low']:
                    buy_points['third_type'].append({
                        'time': data.index[j],
                        'price': data['Low'].iloc[j],
                        'reason': 'Failed test of pivot point low',
                    })
                    break
        for pivot in pivot_points:
            idx = data.index.get_loc(pivot['end'])
            if idx + 5 >= len(data):
                continue
            for j in range(idx + 5, min(idx + 30, len(data))):
                if data['High'].iloc[j] > pivot['high'] * 0.95 and data['High'].iloc[j] < pivot['high']:
                    sell_points['third_type'].append({
                        'time': data.index[j],
                        'price': data['High'].iloc[j],
                        'reason': 'Failed test of pivot point high',
                    })
                    break
        return {'buy': buy_points, 'sell': sell_points}

    # ------------------------------------------------------------------
    # High level analysis and monitoring
    # ------------------------------------------------------------------
    def analyze_stock(self, stock_info, period: str = 'daily'):
        data = self.get_stock_data(stock_info, period)
        if data is None or len(data) < 30:
            return None
        data = self.calculate_macd(data)
        data = self.identify_fractal_tops_bottoms(data)
        strokes = self.identify_stroke(data)
        pivot_points = self.identify_pivot_points(strokes)
        divergences = self.detect_divergence(data, strokes)
        signals = self.identify_buy_sell_points(data, pivot_points, divergences)
        result = {
            'stock': stock_info,
            'period': period,
            'last_price': data['Close'].iloc[-1] if len(data) > 0 else None,
            'pivot_points': pivot_points,
            'divergences': divergences,
            'signals': signals,
        }
        valid_start = datetime.datetime.now() - datetime.timedelta(days=3)
        result['signals'] = {
            'buy': {k: [p for p in v if p['time'] > valid_start] for k, v in signals['buy'].items()},
            'sell': {k: [p for p in v if p['time'] > valid_start] for k, v in signals['sell'].items()},
            'start': valid_start,
            'end': datetime.datetime.now(),
        }
        return result

    def print_analysis_results(self, results, max_display: int = 3):
        print(f"\n{'=' * 80}")
        print(f"股票: {results['stock']['name']} ({results['stock']['code']})")
        print(f"周期: {results['period']} | 最新价: {results['last_price']}")
        print(f"信号时间范围: {results['signals']['start']} 至 {results['signals']['end']}")
        print('-' * 80)
        print("\n关键信号摘要：")
        signal_counts = {
            '买点': sum(len(v) for v in results['signals']['buy'].values()),
            '卖点': sum(len(v) for v in results['signals']['sell'].values()),
        }
        print(f"★ 发现 {signal_counts['买点']} 个买入信号 | {signal_counts['卖点']} 个卖出信号")

        def print_signals(signal_type, signals):
            if signals:
                print(f"\n{signal_type}信号（最近{max_display}个）：")
                for s in signals[:max_display]:
                    time_str = s['time'].strftime('%m-%d %H:%M')
                    print(f"→ {time_str} | 价格: {s['price']:.2f} | 类型: {s['type']}")

        all_signals = []
        for stype, points in results['signals']['buy'].items():
            all_signals.extend([{ 'time': p['time'], 'price': p['price'], 'type': f'买点_{stype}' } for p in points])
        for stype, points in results['signals']['sell'].items():
            all_signals.extend([{ 'time': p['time'], 'price': p['price'], 'type': f'卖点_{stype}' } for p in points])
        all_signals.sort(key=lambda x: x['time'], reverse=True)
        print_signals('最新', all_signals[:max_display])
        if results['divergences']:
            print("\n重要提示：")
            for d in results['divergences']:
                print(f"❗ {d['time'].strftime('%m-%d %H:%M')} 发现{d['type']}背离")
        print(f"\n{'=' * 80}")

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------
    def is_trading_day(self):
        now = datetime.datetime.now()
        if now.weekday() >= 5:
            print(f"{now.date()} 非交易日（周末）")
            return False
        return True

    def in_trading_hours(self):
        now = datetime.datetime.now()
        morning_start = datetime.time(9, 25)
        morning_end = datetime.time(11, 30)
        afternoon_start = datetime.time(12, 55)
        afternoon_end = datetime.time(15, 5)
        return (
            (morning_start <= now.time() <= morning_end) or
            (afternoon_start <= now.time() <= afternoon_end)
        )

    def wait_until_next_trading(self):
        while True:
            now = datetime.datetime.now()
            next_morning = datetime.datetime.combine(now.date(), datetime.time(9, 25))
            next_afternoon = datetime.datetime.combine(now.date(), datetime.time(12, 55))
            if now < next_morning:
                wait_seconds = (next_morning - now).total_seconds()
            elif now.time() < datetime.time(12, 55):
                wait_seconds = (next_afternoon - now).total_seconds()
            else:
                wait_seconds = ((next_morning + datetime.timedelta(days=1)) - now).total_seconds()
            print(f"⏳ 下次检测时间：{datetime.datetime.now() + datetime.timedelta(seconds=wait_seconds)}")
            time.sleep(wait_seconds)
            if self.is_trading_day():
                return

    def monitor_stocks(self, interval: int = 300):
        print(f"Starting monitoring at {datetime.datetime.now()}")
        print(f"Monitoring {len(self.stock_list)} stocks with refresh interval of {interval} seconds")
        try:
            while True:
                print(f"\n{'-' * 40}")
                print(f"Refreshing data at {datetime.datetime.now()}")
                for stock in self.stock_list:
                    print(f"\nAnalyzing {stock['name']} ({stock['code']})")
                    daily_results = self.analyze_stock(stock, 'daily')
                    if daily_results:
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
                        if recent_buy or recent_sell:
                            self.print_analysis_results(daily_results)
                    min30_results = self.analyze_stock(stock, '30min')
                    if min30_results:
                        recent_buy = False
                        recent_sell = False
                        for type_name, points in min30_results['signals']['buy'].items():
                            for point in points:
                                if (datetime.datetime.now() - point['time']).total_seconds() <= 6 * 3600:
                                    recent_buy = True
                                    print(f"ALERT: Recent {type_name} buy signal on 30-min chart at {point['time']}")
                                    print(f"       Price: {point['price']:.2f}, Reason: {point['reason']}")
                        for type_name, points in min30_results['signals']['sell'].items():
                            for point in points:
                                if (datetime.datetime.now() - point['time']).total_seconds() <= 6 * 3600:
                                    recent_sell = True
                                    print(f"ALERT: Recent {type_name} sell signal on 30-min chart at {point['time']}")
                                    print(f"       Price: {point['price']:.2f}, Reason: {point['reason']}")
                        if recent_buy or recent_sell:
                            self.print_analysis_results(min30_results)
                print(f"\nSleeping for {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.disconnect()
