#!/bin/bash
# CZSC Stock Monitor - Crontab安装脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAPP_ROOT="$(dirname "$SCRIPT_DIR")"
CRONTAB_FILE="$SCRIPT_DIR/crontab.txt"

echo "========================================"
echo "CZSC Stock Monitor - 安装定时任务"
echo "========================================"
echo ""
echo "项目路径: $WEBAPP_ROOT"
echo "Crontab配置: $CRONTAB_FILE"
echo ""

# 检查crontab文件是否存在
if [ ! -f "$CRONTAB_FILE" ]; then
    echo "❌ 错误: crontab.txt 文件不存在"
    exit 1
fi

# 创建日志目录
mkdir -p "$WEBAPP_ROOT/logs"
echo "✅ 日志目录已创建: $WEBAPP_ROOT/logs"

# 备份现有crontab
if crontab -l > /dev/null 2>&1; then
    BACKUP_FILE="$WEBAPP_ROOT/logs/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
    crontab -l > "$BACKUP_FILE"
    echo "✅ 现有crontab已备份: $BACKUP_FILE"
fi

# 替换路径变量
sed "s|WEBAPP_ROOT=.*|WEBAPP_ROOT=$WEBAPP_ROOT|g" "$CRONTAB_FILE" > /tmp/crontab_install.txt

# 安装新的crontab
crontab /tmp/crontab_install.txt
rm /tmp/crontab_install.txt

echo ""
echo "✅ 定时任务安装成功！"
echo ""
echo "已安装的任务:"
echo "----------------------------------------"
crontab -l | grep -v "^#" | grep -v "^$" || echo "无任务"
echo "----------------------------------------"
echo ""
echo "📅 定时任务说明:"
echo "  • 15:25 - 更新K线数据"
echo "  • 16:50 - 生成推荐信号"
echo "  • 18:20 - 生成追踪数据"
echo "  • 02:00 - 更新股票列表（每周一）"
echo "  • 03:00 - 清理旧日志"
echo "  • */5分钟 - 监控Web服务"
echo ""
echo "📝 查看日志:"
echo "  tail -f $WEBAPP_ROOT/logs/fetch_kline_$(date +%Y%m%d).log"
echo "  tail -f $WEBAPP_ROOT/logs/recommend_$(date +%Y%m%d).log"
echo "  tail -f $WEBAPP_ROOT/logs/tracking_$(date +%Y%m%d).log"
echo ""
echo "🔧 管理命令:"
echo "  查看任务: crontab -l"
echo "  编辑任务: crontab -e"
echo "  删除任务: crontab -r"
echo ""
