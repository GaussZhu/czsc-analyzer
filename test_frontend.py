#!/usr/bin/env python3
"""测试前端数据加载"""
import asyncio
import aiohttp

async def test_api():
    """测试API返回数据"""
    async with aiohttp.ClientSession() as session:
        # 测试推荐API
        async with session.get('http://localhost:8000/api/tracking/recommend') as resp:
            data = await resp.json()
            print("📊 推荐API测试:")
            print(f"  - success: {data.get('success')}")
            if data.get('success') and data.get('data'):
                markets = list(data['data'].keys())
                print(f"  - 市场数量: {len(markets)}")
                for market in markets[:3]:  # 只显示前3个
                    buy_count = len(data['data'][market].get('买点推荐', []))
                    sell_count = len(data['data'][market].get('卖点推荐', []))
                    print(f"  - {market}: {buy_count}买 {sell_count}卖")
            print()
        
        # 测试追踪API
        async with session.get('http://localhost:8000/api/tracking/list') as resp:
            data = await resp.json()
            print("📍 追踪API测试:")
            print(f"  - success: {data.get('success')}")
            if data.get('success') and data.get('data'):
                items = data['data']
                print(f"  - 追踪数量: {len(items)}")
                for item in items[:3]:  # 只显示前3个
                    print(f"  - {item['code']} {item['name']}: {item['current_profit']}%")
            print()

if __name__ == '__main__':
    asyncio.run(test_api())
