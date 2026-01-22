#!/bin/bash
cd /root/stock_monitor_project/webapp
source venv/bin/activate
exec uvicorn web.main:app --host 0.0.0.0 --port 8000
