#!/bin/bash
# CZSC Stock Monitor - 手动运行所有定时任务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAPP_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "CZSC Stock Monitor - 手动运行所有任务"
echo "========================================"
echo ""
echo "项目路径: $WEBAPP_ROOT"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 创建日志目录
mkdir -p "$WEBAPP_ROOT/logs"

# 进入项目目录
cd "$WEBAPP_ROOT"

# 1. 更新K线数据
echo "====== 任务1: 更新K线数据 ======"
venv/bin/python3 scripts/fetch_daily_kline.py 2>&1 | tee "logs/manual_fetch_kline_$(date +%Y%m%d_%H%M%S).log"
echo "✅ K线数据更新完成"
echo ""

# 2. 生成推荐信号
echo "====== 任务2: 生成推荐信号 ======"
venv/bin/python3 scripts/generate_recommendations.py 2>&1 | tee "logs/manual_recommend_$(date +%Y%m%d_%H%M%S).log"
echo "✅ 推荐信号生成完成"
echo ""

# 3. 生成追踪数据
echo "====== 任务3: 生成追踪数据 ======"
venv/bin/python3 scripts/generate_tracking.py 2>&1 | tee "logs/manual_tracking_$(date +%Y%m%d_%H%M%S).log"
echo "✅ 追踪数据生成完成"
echo ""

echo "========================================"
echo "所有任务执行完成！"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📂 生成的文件:"
echo "  • web/static/recommend_result.json"
echo "  • web/static/tracking_result.json"
echo ""
echo "📝 查看日志:"
echo "  ls -lh logs/manual_*.log"
echo "========================================"
