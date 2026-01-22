#!/bin/bash
# 页面诊断脚本

echo "========================================="
echo "CZSC Web页面诊断报告"
echo "========================================="
echo ""

BASE_URL="http://localhost:8000"

# 测试首页
echo "1. 首页 (/):"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/)
SIZE=$(curl -s $BASE_URL/ | wc -c)
echo "   状态码: $STATUS"
echo "   页面大小: $SIZE bytes"
echo ""

# 测试分析入口页
echo "2. 分析入口 (/analysis):"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/analysis)
SIZE=$(curl -s $BASE_URL/analysis | wc -c)
echo "   状态码: $STATUS"
echo "   页面大小: $SIZE bytes"
echo ""

# 测试完整分析页
echo "3. 完整分析 (/analyze/600000):"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/analyze/600000)
SIZE=$(curl -s $BASE_URL/analyze/600000 | wc -c)
HAS_KLINE=$(curl -s $BASE_URL/analyze/600000 | grep -c "const plotData")
HAS_SIGNALS=$(curl -s $BASE_URL/analyze/600000 | grep -c "rawSignals")
echo "   状态码: $STATUS"
echo "   页面大小: $SIZE bytes"
echo "   K线数据: $HAS_KLINE 处"
echo "   信号数据: $HAS_SIGNALS 处"
echo ""

# 测试追踪页
echo "4. 实时追踪 (/tracking):"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/tracking)
SIZE=$(curl -s $BASE_URL/tracking | wc -c)
echo "   状态码: $STATUS"
echo "   页面大小: $SIZE bytes"
echo ""

# 测试推荐页
echo "5. 智能推荐 (/recommend):"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/recommend)
SIZE=$(curl -s $BASE_URL/recommend | wc -c)
echo "   状态码: $STATUS"
echo "   页面大小: $SIZE bytes"
echo ""

# 测试API健康检查
echo "6. API健康检查 (/health):"
HEALTH=$(curl -s $BASE_URL/health)
echo "   响应: $HEALTH"
echo ""

# 测试静态资源
echo "7. 静态资源检查:"
for file in /static/css/main.css /static/css/index.css /static/js/index.js; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL$file)
    echo "   $file: $STATUS"
done
echo ""

echo "========================================="
echo "诊断完成"
echo "========================================="
