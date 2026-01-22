#!/bin/bash

# 启动CZSC股票分析Web应用
# 作者: GaussZhu
# 日期: 2026-01-22

cd /root/stock_monitor_project/webapp

# 激活虚拟环境
source venv/bin/activate

# 停止旧进程
pkill -f "uvicorn web.main:app"
sleep 2

# 启动服务
echo "🚀 启动Web应用..."
nohup uvicorn web.main:app --host 0.0.0.0 --port 8000 > /tmp/web_server.log 2>&1 &

PID=$!
echo "✅ 服务已启动"
echo "PID: $PID"
echo "日志: /tmp/web_server.log"
echo "访问: http://43.143.235.99:8000"

# 等待服务启动
sleep 5

# 测试API
echo ""
echo "📊 测试市场情绪API..."
curl -s "http://localhost:8000/api/market/sentiment" | python3 -m json.tool | head -15

echo ""
echo "✅ 启动完成！"
