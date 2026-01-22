# CZSC v2.1 访问指南

## 🎉 好消息！你的网站已经可以访问了！

---

## 🌐 正确的访问地址

### ✅ 使用域名访问（推荐）

**HTTPS (安全访问)**:
```
https://www.chanquant.com/
```

**HTTP (自动跳转到 HTTPS)**:
```
http://www.chanquant.com/
```

### ❌ 不要使用 IP 访问

```
❌ http://43.143.235.99:8000/  (无法访问)
```

**原因**: 
- Nginx 只配置了域名访问，没有配置 IP 直接访问
- 这是标准的生产环境配置方式

---

## 📋 可用页面列表

| 功能 | 地址 |
|------|------|
| 首页 | https://www.chanquant.com/ |
| API 文档 | https://www.chanquant.com/docs |
| 健康检查 | https://www.chanquant.com/health |
| 分析页面 | https://www.chanquant.com/analysis |
| 追踪页面 | https://www.chanquant.com/tracking |
| 推荐页面 | https://www.chanquant.com/recommend |

---

## 🔧 当前架构

```
[用户浏览器]
    ↓ HTTPS (443) 或 HTTP (80)
[Nginx 反向代理]
    ↓ HTTP (8000)
[Uvicorn (FastAPI 应用)]
```

**优点**:
- ✅ HTTPS 加密连接（更安全）
- ✅ 自动 HTTP 跳转到 HTTPS
- ✅ Nginx 处理静态文件和缓存
- ✅ 专业的生产环境配置

---

## 🔍 问题排查

### 如果域名访问不了

1. **检查域名解析**
   ```bash
   ping www.chanquant.com
   ```
   应该解析到: 43.143.235.99

2. **检查 Nginx 状态**
   ```bash
   systemctl status nginx
   ```
   应该显示: active (running)

3. **检查后端服务**
   ```bash
   ps aux | grep uvicorn | grep 8000
   ```
   应该看到进程在运行

4. **查看 Nginx 日志**
   ```bash
   tail -f /var/log/nginx/error.log
   tail -f /var/log/nginx/access.log
   ```

---

## 💡 如果需要 IP 直接访问

### 方案 A: 通过 Nginx 代理（推荐）

修改 Nginx 配置，添加 IP 访问支持。

1. **创建新配置文件**
   ```bash
   sudo nano /etc/nginx/sites-available/ip_access
   ```

2. **添加配置**
   ```nginx
   server {
       listen 80;
       server_name 43.143.235.99;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```

3. **启用配置**
   ```bash
   sudo ln -s /etc/nginx/sites-available/ip_access /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

4. **访问**: http://43.143.235.99/

### 方案 B: 直接暴露端口（不推荐）

需要:
1. 云服务商安全组开放 8000 端口
2. 直接访问 http://43.143.235.99:8000/

**缺点**:
- 没有 HTTPS 加密
- 绕过了 Nginx 的安全防护
- 不是标准的生产环境做法

---

## 🔐 安全建议

1. **始终使用 HTTPS**
   - 你的域名已经配置了 Let's Encrypt SSL 证书
   - 访问时会自动跳转到 HTTPS

2. **保持证书更新**
   ```bash
   # 测试证书自动更新
   sudo certbot renew --dry-run
   ```

3. **定期查看访问日志**
   ```bash
   tail -f /var/log/nginx/access.log
   ```

---

## 🎯 测试访问

### 使用浏览器
直接打开: https://www.chanquant.com/

### 使用命令行
```bash
# 测试主页
curl https://www.chanquant.com/

# 测试健康检查
curl https://www.chanquant.com/health

# 测试 API
curl https://www.chanquant.com/api/stock/list
```

---

## 📊 Nginx 配置信息

**主配置**: `/etc/nginx/nginx.conf`  
**站点配置**: `/etc/nginx/sites-available/stock_monitor`  
**启用链接**: `/etc/nginx/sites-enabled/stock_monitor`  

**当前配置**:
- 监听端口: 80 (HTTP), 443 (HTTPS)
- 域名: chanquant.com, www.chanquant.com
- 反向代理: → http://127.0.0.1:8000
- SSL 证书: Let's Encrypt

---

## 🚀 服务管理

### 启动服务
```bash
# 启动 Nginx
sudo systemctl start nginx

# 启动 Web 应用（如果停止了）
cd /root/stock_monitor_project/webapp
source venv/bin/activate
uvicorn web.main:app --host 0.0.0.0 --port 8000 &
```

### 停止服务
```bash
# 停止 Nginx
sudo systemctl stop nginx

# 停止 Web 应用
pkill -f "uvicorn.*8000"
```

### 重启服务
```bash
# 重启 Nginx
sudo systemctl restart nginx

# 重启 Web 应用
pkill -f "uvicorn.*8000"
cd /root/stock_monitor_project/webapp
source venv/bin/activate
uvicorn web.main:app --host 0.0.0.0 --port 8000 &
```

---

## 📞 需要帮助？

如果访问时遇到问题:

1. 检查域名是否正确: www.chanquant.com ✓
2. 确认使用 https:// 而不是 http://
3. 清除浏览器缓存 (Ctrl+F5)
4. 检查服务状态 (Nginx 和 Uvicorn)
5. 查看日志文件

---

**更新时间**: 2026-01-09  
**版本**: CZSC v2.1.0  
**域名**: www.chanquant.com  
**服务器**: 43.143.235.99
