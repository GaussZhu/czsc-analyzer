# CZSC v2.1 最终访问指南

## 🎉 服务已完全修复并上线！

所有页面现已正常工作，可以通过域名访问完整功能。

---

## 🌐 访问地址

### 主要入口
- **主页（登录）**：https://www.chanquant.com/
- **API文档**：https://www.chanquant.com/docs

### 功能页面
- **股票分析**：https://www.chanquant.com/analysis
- **实时追踪**：https://www.chanquant.com/tracking
- **智能推荐**：https://www.chanquant.com/recommend
- **健康检查**：https://www.chanquant.com/health

---

## ✅ 修复完成确认

### 问题回顾
1. **初始问题**：主页可访问，但所有子页面报 HTTP 503/500 错误
2. **原因诊断**：
   - 端口问题：服务运行在3333，但外网只开放8000
   - 模板问题：期望后端数据但路由未提供
3. **解决方案**：
   - ✅ 服务切换到8000端口
   - ✅ 重构页面为前后端分离架构
   - ✅ 使用JavaScript调用API动态加载数据

### 当前状态
| 组件 | 状态 | 说明 |
|------|------|------|
| Web服务 | ✅ 运行中 | 端口8000，PID: 2108062 |
| Nginx反向代理 | ✅ 正常 | chanquant.com → localhost:8000 |
| SSL证书 | ✅ 有效 | Let's Encrypt |
| 主页 | ✅ 正常 | https://www.chanquant.com/ |
| 分析页面 | ✅ 正常 | https://www.chanquant.com/analysis |
| 追踪页面 | ✅ 正常 | https://www.chanquant.com/tracking |
| 推荐页面 | ✅ 正常 | https://www.chanquant.com/recommend |
| API文档 | ✅ 正常 | https://www.chanquant.com/docs |

---

## 📖 使用指南

### 1. 股票分析 (/analysis)
**功能**：输入股票代码进行技术分析

**使用步骤**：
1. 访问 https://www.chanquant.com/analysis
2. 在输入框输入股票代码（如：600000）
3. 点击"分析"按钮或按回车键
4. 查看分析结果（JSON格式）

**API调用**：
```javascript
GET /api/analysis/quick/{code}
```

### 2. 实时追踪 (/tracking)
**功能**：查看正在追踪的股票列表

**使用步骤**：
1. 访问 https://www.chanquant.com/tracking
2. 页面自动加载追踪列表
3. 查看追踪状态和数据

**API调用**：
```javascript
GET /api/tracking/list
```

### 3. 智能推荐 (/recommend)
**功能**：查看系统推荐的股票

**使用步骤**：
1. 访问 https://www.chanquant.com/recommend
2. 页面自动加载推荐结果
3. 查看推荐理由和数据

**API调用**：
```javascript
GET /api/tracking/recommend
```

---

## 🔧 技术架构

### 前端架构
```
用户浏览器
    ↓
HTTPS (443) - SSL加密
    ↓
Nginx 反向代理 (chanquant.com)
    ↓
HTTP (8000) - 内网
    ↓
FastAPI应用 (CZSC v2.1)
    ↓
MySQL数据库 (Unix Socket)
```

### 页面加载流程
```
1. 用户访问页面 (HTML模板)
2. JavaScript执行
3. fetch() 调用 API
4. 后端处理请求
5. 返回JSON数据
6. 前端动态渲染
```

---

## 🚀 API端点列表

### 股票相关
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/stock/list` | GET | 获取股票列表 |
| `/api/stock/search?q={query}` | GET | 搜索股票 |
| `/api/stock/info/{code}` | GET | 获取股票信息 |

### 分析相关
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/analysis/quick/{code}` | GET | 快速分析 |
| `/api/analysis/analyze` | POST | 完整分析 |
| `/api/analysis/history/{code}` | GET | 历史分析 |

### 追踪相关
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/tracking/list` | GET | 追踪列表 |
| `/api/tracking/add` | POST | 添加追踪 |
| `/api/tracking/remove/{code}` | DELETE | 移除追踪 |
| `/api/tracking/recommend` | GET | 推荐列表 |

完整API文档：https://www.chanquant.com/docs

---

## 🔍 测试方法

### 浏览器测试
```
1. 打开 https://www.chanquant.com/
2. 点击导航栏各个菜单
3. 测试各项功能
```

### 命令行测试
```bash
# 健康检查
curl https://www.chanquant.com/health

# 获取股票列表
curl https://www.chanquant.com/api/stock/list

# 快速分析
curl https://www.chanquant.com/api/analysis/quick/600000

# 追踪列表
curl https://www.chanquant.com/api/tracking/list

# 推荐列表
curl https://www.chanquant.com/api/tracking/recommend
```

### PowerShell测试（Windows）
```powershell
# 健康检查
Invoke-WebRequest -Uri https://www.chanquant.com/health

# 获取API数据
(Invoke-WebRequest -Uri https://www.chanquant.com/api/stock/list).Content | ConvertFrom-Json
```

---

## 📊 服务管理

### 查看服务状态
```bash
# 检查端口
netstat -tulpn | grep :8000

# 查看进程
ps aux | grep uvicorn | grep 8000

# 健康检查
curl http://localhost:8000/health
```

### 查看日志
```bash
# Web服务日志
tail -f /tmp/czsc_web_v21_port8000.log

# Nginx访问日志
tail -f /var/log/nginx/access.log

# Nginx错误日志
tail -f /var/log/nginx/error.log
```

### 重启服务
```bash
# 停止服务
pkill -f "uvicorn.*8000"

# 启动服务
cd /root/stock_monitor_project/webapp
source venv/bin/activate
nohup uvicorn web.main:app --host 0.0.0.0 --port 8000 --log-level info > /tmp/czsc_web_v21_port8000.log 2>&1 &
```

### 重启Nginx
```bash
# 测试配置
sudo nginx -t

# 重启Nginx
sudo systemctl restart nginx

# 查看状态
sudo systemctl status nginx
```

---

## 📈 项目统计

### v2.0 后端重构
- **Python模块**：25 个
- **代码行数**：5000+ 行
- **数据表**：7 张 (MySQL)
- **连接方式**：Unix Socket (/var/run/mysqld/mysqld.sock)
- **架构**：完全模块化 (DAO, Service, Core, Config)

### v2.1 Web重构
- **Python模块**：13 个
- **代码行数**：4300+ 行
- **API路由**：23 个
- **页面数量**：6 个
- **架构**：前后端分离，API驱动

### 整体统计
- **总文件数**：80+ 文件
- **总代码量**：9000+ 行代码
- **Git提交**：19 次提交
- **文档数量**：13 份文档

---

## 🎯 最终测试结果

### ✅ 所有测试通过
- [x] Web服务启动正常
- [x] 健康检查通过
- [x] 主页加载正常
- [x] 分析页面正常（输入框+API调用）
- [x] 追踪页面正常（动态加载）
- [x] 推荐页面正常（动态加载）
- [x] API文档可访问
- [x] SSL证书有效
- [x] Nginx反向代理正常
- [x] MySQL连接正常

### 访问确认
```bash
$ curl https://www.chanquant.com/health
{"status":"healthy","version":"2.1.0","service":"CZSC Stock Analyzer"}

$ curl -I https://www.chanquant.com/
HTTP/1.1 200 OK
Server: nginx
Content-Type: text/html; charset=utf-8
```

---

## 🎉 总结

**CZSC v2.1 股票分析工具已完全部署成功！**

### 关键成就
1. ✅ **问题诊断**：快速定位端口和模板问题
2. ✅ **架构重构**：前后端分离，API驱动
3. ✅ **全面测试**：本地+外网测试通过
4. ✅ **生产就绪**：SSL、Nginx、后台服务
5. ✅ **文档完善**：13份技术文档

### 可用功能
- 🏠 主页展示
- 📊 股票分析（输入代码查询）
- 📍 实时追踪（动态加载列表）
- 💡 智能推荐（算法推荐）
- 📖 完整API文档
- 🔒 HTTPS安全访问

### 下一步
所有功能已就绪，可以正常使用！

如有问题，请查看：
- 服务日志：`/tmp/czsc_web_v21_port8000.log`
- Nginx日志：`/var/log/nginx/access.log`
- 错误日志：`/var/log/nginx/error.log`

---

**项目信息**
- **名称**：CZSC 股票分析工具
- **版本**：v2.1.0
- **域名**：https://www.chanquant.com/
- **GitHub**：https://github.com/GaussZhu/czsc-analyzer
- **分支**：genspark_ai_developer
- **联系**：zhu_gs@126.com

---

*文档生成时间：2026-01-09*  
*状态：生产环境运行中 ✅*
