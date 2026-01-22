#!/usr/bin/env python3
"""
测试所有页面的加载和数据显示
"""
import requests
import json
from bs4 import BeautifulSoup
import re

def test_page(url, page_name):
    """测试单个页面"""
    print(f"\n{'='*60}")
    print(f"测试页面: {page_name}")
    print(f"URL: {url}")
    print('='*60)
    
    try:
        response = requests.get(url, timeout=10)
        print(f"✓ HTTP状态码: {response.status_code}")
        print(f"✓ 页面大小: {len(response.text)} 字节")
        
        # 检查是否包含Plotly
        if 'plotly' in response.text.lower():
            print("✓ 包含Plotly库")
        
        # 检查K线图数据
        if 'plotData' in response.text:
            # 提取plotData
            match = re.search(r'const plotData\s*=\s*(\{.+?\});', response.text, re.DOTALL)
            if match:
                try:
                    # 简单检查数据结构
                    data_str = match.group(1)
                    if '"data"' in data_str and '"layout"' in data_str:
                        print("✓ K线图数据结构完整")
                        # 检查是否有实际数据
                        if '"x"' in data_str and '"open"' in data_str:
                            print("✓ K线数据点存在")
                        else:
                            print("✗ K线数据为空")
                    else:
                        print("✗ K线图数据结构不完整")
                except Exception as e:
                    print(f"✗ 解析K线图数据失败: {e}")
            else:
                print("✗ 未找到plotData变量")
        
        # 检查买卖点数据
        if 'rawSignals' in response.text:
            match = re.search(r'rawSignals\s*=\s*JSON\.parse\(\'(.+?)\'\)', response.text)
            if match:
                try:
                    signals_str = match.group(1)
                    # 解码Unicode转义
                    signals_str = signals_str.encode().decode('unicode_escape')
                    signals = json.loads(signals_str)
                    buy_count = sum(1 for s in signals if '买点' in s.get('type', ''))
                    sell_count = sum(1 for s in signals if '卖点' in s.get('type', ''))
                    print(f"✓ 买卖点信号: {buy_count}个买点, {sell_count}个卖点")
                except Exception as e:
                    print(f"✗ 解析买卖点数据失败: {e}")
            else:
                print("✗ 未找到rawSignals数据")
        
        # 检查是否有错误信息
        if 'error' in response.text.lower() or '错误' in response.text:
            soup = BeautifulSoup(response.text, 'html.parser')
            error_divs = soup.find_all(string=re.compile('错误|error', re.I))
            if error_divs:
                print(f"⚠ 页面包含错误信息: {error_divs[:2]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    base_url = "http://localhost:8000"
    
    # 测试各个页面
    pages = [
        (f"{base_url}/", "主页"),
        (f"{base_url}/analysis", "分析入口页"),
        (f"{base_url}/analyze/600000", "完整分析页(600000)"),
        (f"{base_url}/analyze/000001", "完整分析页(000001)"),
        (f"{base_url}/tracking", "实时追踪页"),
        (f"{base_url}/recommend", "智能推荐页"),
    ]
    
    results = []
    for url, name in pages:
        result = test_page(url, name)
        results.append((name, result))
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print('='*60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    # 统计
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 个页面测试通过")

if __name__ == "__main__":
    main()
