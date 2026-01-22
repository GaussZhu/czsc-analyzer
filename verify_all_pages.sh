#!/bin/bash
# 验证所有页面和API的完整性

echo "========================================="
echo "📊 CZSC股票分析系统 - 页面验证"
echo "========================================="
echo ""

BASE_URL="http://localhost:8000"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试API
echo "1️⃣  测试API接口..."
echo "-------------------------------------------"

# 推荐API
echo -n "   /api/tracking/recommend ... "
RESPONSE=$(curl -s "${BASE_URL}/api/tracking/recommend")
SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)
if [ "$SUCCESS" = "True" ]; then
    BUY_COUNT=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {})
count = sum(len(m.get('买点推荐', [])) for m in data.values())
print(count)
" 2>/dev/null)
    echo -e "${GREEN}✓${NC} (${BUY_COUNT}个买点)"
else
    echo -e "${RED}✗${NC}"
fi

# 追踪API  
echo -n "   /api/tracking/list ... "
RESPONSE=$(curl -s "${BASE_URL}/api/tracking/list")
SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)
if [ "$SUCCESS" = "True" ]; then
    COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data', [])))" 2>/dev/null)
    echo -e "${GREEN}✓${NC} (${COUNT}个追踪)"
else
    echo -e "${RED}✗${NC}"
fi

echo ""

# 测试页面
echo "2️⃣  测试前端页面..."
echo "-------------------------------------------"

check_page() {
    local path=$1
    local name=$2
    local keyword=$3
    
    echo -n "   ${name} (${path}) ... "
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${path}")
    
    if [ "$HTTP_CODE" = "200" ]; then
        if [ -n "$keyword" ]; then
            CONTENT=$(curl -s "${BASE_URL}${path}")
            if echo "$CONTENT" | grep -q "$keyword"; then
                echo -e "${GREEN}✓${NC} (HTTP $HTTP_CODE)"
            else
                echo -e "${YELLOW}⚠${NC} (HTTP $HTTP_CODE, 缺少关键词)"
            fi
        else
            echo -e "${GREEN}✓${NC} (HTTP $HTTP_CODE)"
        fi
    else
        echo -e "${RED}✗${NC} (HTTP $HTTP_CODE)"
    fi
}

check_page "/" "首页" "CZSC"
check_page "/analysis" "股票分析" "股票分析"
check_page "/recommend" "智能推荐" "智能推荐"
check_page "/tracking" "实时追踪" "实时追踪"
check_page "/analyze/600000" "股票详情" "600000"

echo ""

# 测试静态资源
echo "3️⃣  测试静态资源..."
echo "-------------------------------------------"

check_static() {
    local path=$1
    local name=$2
    
    echo -n "   ${name} ... "
    if [ -f "web/static/${path}" ]; then
        SIZE=$(stat -f%z "web/static/${path}" 2>/dev/null || stat -c%s "web/static/${path}" 2>/dev/null)
        if [ "$SIZE" -gt 100 ]; then
            echo -e "${GREEN}✓${NC} ($(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo ${SIZE}B))"
        else
            echo -e "${YELLOW}⚠${NC} (文件太小: ${SIZE}B)"
        fi
    else
        echo -e "${RED}✗${NC} (文件不存在)"
    fi
}

check_static "recommend_result.json" "推荐结果数据"
check_static "tracking_result.json" "追踪结果数据"
check_static "buffett_analysis_full_20251204.csv" "基本面数据"

echo ""

# 总结
echo "========================================="
echo "✅ 验证完成！"
echo "========================================="
echo ""
echo "📍 在线访问地址:"
echo "   - 首页: https://www.chanquant.com/"
echo "   - 推荐: https://www.chanquant.com/recommend"
echo "   - 追踪: https://www.chanquant.com/tracking"
echo "   - 分析: https://www.chanquant.com/analyze/600000"
echo ""
