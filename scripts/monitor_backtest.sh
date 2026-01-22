#!/bin/bash
# 回测进度监控脚本

echo "=========================================="
echo "回测进度监控"
echo "=========================================="
echo ""

# 检查进程是否在运行
if ps aux | grep "backtest_full.py" | grep -v grep > /dev/null; then
    echo "✅ 回测任务正在运行中..."
    echo ""
else
    echo "❌ 没有发现运行中的回测任务"
    echo ""
fi

# 显示最新进度
echo "📊 最新进度："
echo "------------------------------------------"
tail -100 /tmp/backtest_500.log 2>/dev/null | grep -E "(INFO.*backtest_full|进度|分析完成|推荐数|市场|胜率|收益)" | tail -20
echo ""

# 显示统计信息
echo "📈 统计信息："
echo "------------------------------------------"
total_analyzed=$(grep -c "开始分析股票" /tmp/backtest_500.log 2>/dev/null || echo "0")
echo "已分析股票数: $total_analyzed"

if grep -q "分析完成" /tmp/backtest_500.log 2>/dev/null; then
    echo "✅ 第一阶段完成（生成推荐）"
fi

if grep -q "胜率" /tmp/backtest_500.log 2>/dev/null; then
    echo "✅ 第二阶段完成（计算收益）"
fi

if grep -q "回测完成" /tmp/backtest_500.log 2>/dev/null; then
    echo "🎉 回测全部完成！"
fi

echo ""
echo "=========================================="
echo "实时监控命令:"
echo "tail -f /tmp/backtest_500.log | grep -E '(进度|推荐数|胜率)'"
echo "=========================================="
