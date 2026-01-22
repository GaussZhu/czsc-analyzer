import sys
sys.path.insert(0, '/root/stock_monitor_project/webapp')

from app.core.data_source import UnifiedDataSource, DataSourceType

# 初始化数据源
data_source = UnifiedDataSource(DataSourceType.MYSQL)

# 获取002314的日线数据
code = '002314'
market = 0  # 深市
period = 'daily'

print(f"获取 {code} 的 {period} 数据...")
data = data_source.get_stock_data(code, market, period, prefer_mysql=True)

if data is not None:
    print(f"✅ 数据获取成功！")
    print(f"数据条数: {len(data)}")
    print(f"最新日期: {data.index[-1]}")
    print(f"最早日期: {data.index[0]}")
    print(f"\n最近5条数据:")
    print(data.tail(5))
else:
    print("❌ 数据获取失败")
