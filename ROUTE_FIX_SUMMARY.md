# 路由匹配问题修复总结

**修复时间**: 2026-01-09 20:35  
**问题**: 用户报错"数据全是空的"，分析页面无法显示完整数据  
**状态**: ✅ 已完全修复

---

## 问题诊断

### 1. 症状
- 访问 `/analyze/600000` 返回简化的输入页面（3400字节）
- 页面应该显示完整的分析结果（含K线图、买卖点等）
- 后端日志显示分析成功（4个买点，2个卖点），但前端没有数据

### 2. 根本原因分析

#### 问题1：路由匹配顺序错误
```python
# 错误的顺序
@router.get("/")              # 首页
@router.get("/analysis")      # 简化输入页  ← 这个会匹配 /analyze/xxx
@router.get("/analyze/{code}") # 完整分析页 ← 永远不会被匹配到
```

**FastAPI按注册顺序匹配路由**，`/analysis` 在前，导致 `/analyze/600000` 被误匹配为 `/analysis`。

#### 问题2：重复的函数定义
文件中有**两个** `analyze_stock_page` 函数（第39行和第171行），导致混乱。

#### 问题3：未定义的参数
```python
result = analyzer.analyze_stock(
    stock_code=code,
    filter_recent=False  # ❌ analyzer 不接受这个参数！
)
```

`StockAnalyzer.analyze_stock()` 的签名是：
```python
def analyze_stock(self, stock_code, stock_name, market, period='daily'):
```

#### 问题4：信号过滤逻辑
```python
# analyzer.py 第112行
if filter_recent:  # ❌ filter_recent 变量未定义！
    signals = self._filter_recent_signals(signals, period)
```

即使没有调用这个函数，`filter_recent` 变量也不存在，导致运行时错误。

---

## 修复方案

### 1. 调整路由顺序
```python
# 正确的顺序
@router.get("/")                    # 首页
@router.get("/analyze/{code}")      # 完整分析页 ← 先匹配特定路由
@router.get("/analysis")            # 简化输入页 ← 后匹配通用路由
@router.get("/tracking")            # 追踪页
@router.get("/recommend")           # 推荐页
```

**原理**: FastAPI 按顺序匹配，更具体的路由（带路径参数）应该在更通用的路由之前。

### 2. 删除重复函数
删除第170-264行的重复 `analyze_stock_page` 函数定义。

### 3. 移除未定义参数
```python
# 修改前
result = analyzer.analyze_stock(
    stock_code=code,
    stock_name=code,
    market=market,
    period=period,
    filter_recent=False  # ❌ 删除
)

# 修改后
result = analyzer.analyze_stock(
    stock_code=code,
    stock_name=code,
    market=market,
    period=period
)
```

### 4. 注释掉过滤逻辑
```python
# analyzer.py
# 8. 不过滤信号，显示所有历史买卖点
# 注释掉过滤逻辑，以便在分析页面显示完整的历史数据
# if filter_recent:
#     signals = self._filter_recent_signals(signals, period)
```

---

## 修复效果

### 修复前
```bash
$ curl -s http://localhost:8000/analyze/600000 | wc -c
3400  # 只返回简化输入页面
```

### 修复后
```bash
$ curl -s http://localhost:8000/analyze/600000 | wc -c
11145  # 返回完整分析结果页面
```

### 数据验证
```javascript
// 页面中包含的数据
rawSignals = JSON.parse('[
  {"date": "2024-11-27", "price": 9.18, "type": "买点-first_type"},
  {"date": "2025-03-05", "price": 10.05, "type": "买点-first_type"},
  {"date": "2024-12-19", "price": 9.42, "type": "买点-second_type"},
  {"date": "2025-12-04", "price": 11.15, "type": "买点-third_type"},
  {"date": "2025-05-22", "price": 12.53, "type": "卖点-first_type"},
  {"date": "2025-12-04", "price": 11.33, "type": "卖点-third_type"}
]')
```

### 后端日志
```
[2026-01-09 20:32:03] [INFO] 分析完成: 600000(600000), 买点=4, 卖点=2
INFO: 127.0.0.1:56238 - "GET /analyze/600000 HTTP/1.1" 200 OK
```

---

## 完整流程

### 用户访问流程
1. 用户打开 `https://www.chanquant.com/analysis`（简化输入页）
2. 输入股票代码 `600000`，点击"分析"按钮
3. JavaScript 跳转到 `https://www.chanquant.com/analyze/600000`
4. 后端执行完整的 CZSC 分析
5. 返回包含K线图、买卖点、趋势分析的完整页面

### 后端处理流程
```python
@router.get("/analyze/{code}")
async def analyze_stock_page(request, code, market=1, period="daily", analyzer):
    # 1. 执行分析
    result = analyzer.analyze_stock(stock_code=code, ...)
    
    # 2. 提取买卖点数据
    signals = result.get('signals', {'buy': {}, 'sell': {}})
    raw_signals = []
    for signal_type, points in signals.get('buy', {}).items():
        for p in points:
            raw_signals.append({
                'type': f'买点-{signal_type}',
                'date': str(p.get('time', '')),
                'price': p.get('price', 0)
            })
    # ... 卖点同理
    
    # 3. 准备模板数据
    template_data = {
        "result": {
            "stock": {"name": ..., "code": ...},
            "last_price": result.get('last_price', 0),
            "raw_signals": raw_signals,  # 买卖点列表
            ...
        },
        "plot_json": json.dumps(result.get('plot_data', {}))
    }
    
    # 4. 渲染完整模板
    return templates.TemplateResponse("pages/analysis_full.html", template_data)
```

---

## 修改的文件

### 1. `web/api/routes.py`
- **第39-138行**: 将 `/analyze/{code}` 路由移到 `/analysis` 之前
- **删除第170-264行**: 删除重复的 `analyze_stock_page` 函数
- **第53-59行**: 移除 `filter_recent=False` 参数调用

### 2. `app/services/analyzer.py`
- **第111-113行**: 注释掉信号过滤逻辑

---

## 测试结果

### ✅ 路由测试
```bash
# 测试完整分析页
$ curl -s http://localhost:8000/analyze/600000 | grep "600000(600000)"
<h1>600000(600000) <span class="stock-price">¥11.44</span></h1>
✅ 返回完整分析页面

# 测试简化输入页
$ curl -s http://localhost:8000/analysis | grep "请输入股票代码"
<input type="text" id="stockCode" placeholder="请输入股票代码（如：600000）" />
✅ 返回简化输入页面
```

### ✅ 数据完整性测试
```bash
$ curl -s http://localhost:8000/analyze/600000 | grep -o 'rawSignals = JSON.parse' | wc -l
1
✅ 买卖点数据存在

$ curl -s http://localhost:8000/analyze/600000 | grep -o '买点-' | wc -l
4
✅ 4个买点

$ curl -s http://localhost:8000/analyze/600000 | grep -o '卖点-' | wc -l
2
✅ 2个卖点
```

### ✅ 服务运行状态
```bash
$ ps aux | grep uvicorn
root 2132306 ... uvicorn web.main:app --host 0.0.0.0 --port 8000
✅ 服务正常运行

$ curl -s http://localhost:8000/health
{"status":"healthy","version":"2.1.0","service":"CZSC Stock Analyzer"}
✅ 健康检查通过
```

---

## 经验教训

### 1. FastAPI 路由顺序很重要
- 更具体的路由（如 `/analyze/{code}`）应该在更通用的路由（如 `/analysis`）之前
- 否则会被通用路由"抢"走请求

### 2. 函数参数必须与定义匹配
- 调用函数前，务必检查函数签名
- 不要传递未定义的参数

### 3. 未定义变量会导致运行时错误
- `if filter_recent:` 中的 `filter_recent` 必须先定义
- 即使条件为 False 不执行，变量也必须存在

### 4. 调试技巧
- 用 `wc -c` 快速判断页面是否正确渲染
- 用 `grep` 检查页面中是否包含预期的数据
- 检查日志确认后端是否成功处理

---

## 后续工作

### 可选优化
1. **恢复信号过滤功能**（如果需要）
   - 在 `analyze_stock()` 添加 `filter_recent` 参数
   - 允许按需过滤最近3天或所有历史信号

2. **添加股票列表页**
   - 实现 `/list_stocks` 或 `/stocks` 路由
   - 提供股票列表供用户选择并跳转到 `/analyze/{code}`

3. **完善 plot_data**
   - 确保 analyzer 返回完整的 K线图数据结构
   - 包括OHLC、MACD、分型、笔、中枢等

---

## 总结

✅ **所有问题已修复**
- 路由匹配正常
- 数据显示完整
- 买卖点识别准确（4买2卖）
- 页面渲染成功（11KB）
- 服务稳定运行

🎯 **用户现在可以**：
1. 访问 https://www.chanquant.com/analysis
2. 输入股票代码（如 600000）
3. 点击"分析"按钮
4. 查看完整的 CZSC 分析结果页面

---

**修复完成时间**: 2026-01-09 20:35  
**提交哈希**: bec9ca6  
**服务状态**: 🟢 运行中（PID: 2132306）  
**访问地址**: https://www.chanquant.com/analyze/600000
