import sys
sys.path.insert(0, '/root/stock_monitor_project/webapp')

from app.services.analyzer import StockAnalyzer

# 初始化analyzer
print("初始化StockAnalyzer...")
analyzer = StockAnalyzer(use_mysql=True)

code = '002314'
market = 0
period = 'daily'

# 1. 测试analyzer内部获取的数据
print(f"\n1. 测试analyzer.analyze_stock获取的数据...")
result = analyzer.analyze_stock(code, code, market, period)

if result:
    print(f"✅ 分析结果获取成功")
    print(f"last_price: {result.get('last_price')}")
    print(f"last_update: {result.get('last_update')}")
    print(f"data_count: {result.get('data_count')}")

# 2. 测试直接从data_source获取
print(f"\n2. 测试analyzer.data_source.get_stock_data...")
kline = analyzer.data_source.get_stock_data(code, market, period, prefer_mysql=True)

if kline is not None:
    print(f"✅ K线数据获取成功")
    print(f"数据条数: {len(kline)}")
    print(f"最新日期: {kline.index[-1]}")
    print(f"最早日期: {kline.index[0]}")
    print(f"\n最近10条:")
    print(kline.tail(10)[['Open', 'High', 'Low', 'Close']])
else:
    print("❌ K线数据为None")

# 3. 测试不带prefer_mysql参数
print(f"\n3. 测试不带prefer_mysql参数...")
kline2 = analyzer.data_source.get_stock_data(code, market, period)

if kline2 is not None:
    print(f"✅ K线数据获取成功")
    print(f"数据条数: {len(kline2)}")
    print(f"最新日期: {kline2.index[-1]}")
else:
    print("❌ K线数据为None")
    
analyzer.data_source.disconnect()
