# CZSC Stock Analysis Tool (缠中说禅股票分析工具)

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.6%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A powerful stock analysis tool based on the Cheng Zhong Shuo Chan (CZSC) technical analysis theory (缠中说禅技术分析理论). This tool monitors stocks on the Chinese A-share market for buy/sell signals on daily and 30-minute timeframes.

## 📋 Features

- **Data Retrieval**: Uses pytdx to fetch historical and real-time stock data
- **CZSC Analysis**: Implements core concepts from CZSC theory:
  - Pivot Point (中枢) Identification
  - Trend Analysis (趋势分析)
  - Divergence Detection (背驰识别)
  - Three Types of Buy/Sell Signals (三类买卖点)
- **Intelligent Monitoring**: Automatically runs during trading hours and sleeps during non-trading periods
- **Server Optimization**: Automatically selects the best TDX server for reliable data access
- **Alert System**: Provides clear notifications when important signals are detected

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/czsc-stock-analysis.git
cd czsc-stock-analysis
```

2. Install the required dependencies:
```bash
pip install pytdx pandas numpy matplotlib
```

## 🚀 Quick Start

1. Edit the `stock_list` in the script to include the stocks you want to monitor:
```python
self.stock_list = [
    {'code': '600000', 'name': '浦发银行', 'market': TDXParams.MARKET_SH},
    {'code': '600004', 'name': '白云机场', 'market': TDXParams.MARKET_SH},
    # Add more stocks as needed
]
```

2. Run the script to start monitoring:
```bash
python CZSC_Stock_Analysis_Tool.py
```

3. The script will automatically:
   - Connect to the best available TDX server
   - Check if it's currently a trading day/hour
   - Analyze your stocks for buy/sell signals
   - Display alerts when significant signals are detected

## 📊 Using the Analysis Tool

### Continuous Monitoring

The default mode automatically monitors your stock list and alerts you when it identifies buy/sell signals:

```python
# Run continuous monitoring (checks every 5 minutes)
analyzer.monitor_stocks(interval=300)
```

### Individual Stock Analysis

You can also analyze a specific stock on demand:

```python
# Analyze a specific stock
stock_info = {'code': '600000', 'name': '浦发银行', 'market': TDXParams.MARKET_SH}
daily_results = analyzer.analyze_stock(stock_info, 'daily')
analyzer.print_analysis_results(daily_results)
```

## 📖 CZSC Theory Implementation

The implementation follows key principles from CZSC theory:

### 1. Fractal Analysis (分型分析)
Identifies tops and bottoms in price movements as the foundation for stroke (笔) analysis.

### 2. Stroke Identification (笔的识别)
Connects valid fractal tops and bottoms to form strokes according to CZSC rules.

### 3. Pivot Points (中枢)
Identifies pivot points as areas where at least three consecutive sub-level movement types overlap.

### 4. Three Types of Buy/Sell Points
- **First Type (第一类买卖点)**: Based on trend reversal with divergence
- **Second Type (第二类买卖点)**: Based on retracements after a first type signal
- **Third Type (第三类买卖点)**: Based on failed tests of pivot point extremes

## 🔍 Signal Types

### Buy Signals (买点)
1. **First Type Buy Point (第一类买点)**
   - Occurs at the end of a downtrend with bullish divergence
   - Most reliable for initiating new positions

2. **Second Type Buy Point (第二类买点)**
   - Occurs during a pullback after a first type buy signal
   - Good for adding to existing positions

3. **Third Type Buy Point (第三类买点)**
   - Occurs when price tests but fails to break below a pivot point low
   - Indicates strong upward momentum

### Sell Signals (卖点)
1. **First Type Sell Point (第一类卖点)**
   - Occurs at the end of an uptrend with bearish divergence
   - Signals potential trend reversal

2. **Second Type Sell Point (第二类卖点)**
   - Occurs during a bounce after a first type sell signal
   - Good opportunity to exit or reduce positions

3. **Third Type Sell Point (第三类卖点)**
   - Occurs when price tests but fails to break above a pivot point high
   - Indicates strong downward momentum

## ⚙️ Customization

You can customize various aspects of the script:

1. **Stocks to monitor**: Edit the `stock_list` variable
2. **Monitoring interval**: Change the `interval` parameter in `monitor_stocks()`
3. **Technical parameters**: Adjust MACD parameters, fractal detection settings, etc.
4. **Alert criteria**: Modify the conditions in the monitoring loop

## 📝 Notes

- The script requires a stable internet connection to access TDX servers
- Performance may vary depending on the reliability of the data source
- For best results, run the script during market hours on trading days

## 🛠 Advanced Usage

### Adding Custom Signal Filters

You can add additional filters to the buy/sell signals to increase accuracy:

```python
# Example: Add volume confirmation to buy signals
if data['Volume'].iloc[signal_index] > data['Volume'].iloc[signal_index-1] * 1.5:
    # Volume increased by 50% - stronger confirmation
    buy_confidence = "High"
else:
    buy_confidence = "Normal"
```

### Customizing Monitoring Schedule

Adjust the trading hours detection to match your specific needs:

```python
def in_trading_hours(self):
    """Custom trading hours definition"""
    now = datetime.datetime.now()
    # Example: Only monitor in the afternoon session
    afternoon_start = datetime.time(13, 0)
    afternoon_end = datetime.time(15, 0)
    return afternoon_start <= now.time() <= afternoon_end
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- This tool is based on the technical analysis theory developed by "Cheng Zhong Shuo Chan" (缠中说禅)
- Thanks to the pytdx library for providing access to stock market data

---

## 📬 Contact

For questions, issues, or contributions, please open an issue on GitHub or contact me at:
[your-email@example.com](mailto:zhu_gs@126.com)

---