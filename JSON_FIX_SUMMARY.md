# K线图JSON序列化问题修复

## 问题发现

用户反馈："还是不对，K线图没有出来"

## 根本原因

**双重JSON转义问题**

### 错误的代码流程

1. **routes.py**:
```python
plot_json": json.dumps(plot_data)  # 第一次JSON序列化
```

2. **模板 analysis_full.html**:
```javascript
const plotData = JSON.parse('{{ plot_json | tojson | safe }}');  
// 第二次JSON序列化！
```

3. **生成的HTML**:
```javascript
const plotData = JSON.parse('"{\"data\": [...]}}"');
// ❌ 双重转义！外层有引号
```

4. **JavaScript解析结果**:
```javascript
plotData = "{\"data\": [...]}"  // 是一个字符串，不是对象！
plotData.data  // undefined
```

### 为什么K线图不显示

```javascript
Plotly.newPlot('kline-chart', plotData.data, plotData.layout);
// plotData.data = undefined (因为plotData是字符串)
// Plotly无法渲染
```

## 修复方案

### 正确的代码流程

1. **routes.py**:
```python
"plot_data": plot_data  # 直接传递字典
```

2. **模板 analysis_full.html**:
```javascript
const plotData = {{ plot_data | tojson | safe }};
// 只进行一次JSON序列化
```

3. **生成的HTML**:
```javascript
const plotData = {"data": [...]};
// ✅ 正确！直接是对象
```

4. **JavaScript解析结果**:
```javascript
plotData = {data: [...], layout: {...}}  // 是对象
plotData.data  // 数组
```

## 修改的文件

### 1. web/api/routes.py (第182行)

**修改前**:
```python
"plot_json": json.dumps(plot_data) if plot_data['data'] else '{}'
```

**修改后**:
```python
"plot_data": plot_data  # 直接传递字典，让模板处理JSON序列化
```

### 2. web/templates/pages/analysis_full.html (第213-214行)

**修改前**:
```javascript
const plotData = JSON.parse('{{ plot_json | tojson | safe }}');
Plotly.newPlot('kline-chart', plotData.data, plotData.layout);
```

**修改后**:
```javascript
const plotData = {{ plot_data | tojson | safe }};
if (plotData && plotData.data && plotData.data.length > 0) {
  Plotly.newPlot('kline-chart', plotData.data, plotData.layout);
} else {
  document.getElementById('kline-placeholder').style.display = 'block';
}
```

## 验证方法

### 1. 查看生成的HTML

**修复前**:
```javascript
const plotData = JSON.parse('"{\\\"data\\\": [...]}}"');
```

**修复后**:
```javascript
const plotData = {"data": [...]};
```

### 2. 浏览器控制台测试

修复前，在浏览器控制台执行：
```javascript
console.log(plotData);  // 显示字符串 "{\"data\": [...]}"
console.log(plotData.data);  // undefined
```

修复后：
```javascript
console.log(plotData);  // 显示对象 {data: Array(1), layout: {...}}
console.log(plotData.data);  // 显示数组 Array(1)
```

## 重启服务

修复已提交到git，现在需要重启服务使修改生效。

### 方法1：使用启动脚本
```bash
cd /root/stock_monitor_project/webapp
pkill -f uvicorn
./start_service.sh
```

### 方法2：直接启动
```bash
cd /root/stock_monitor_project/webapp
pkill -f uvicorn
source venv/bin/activate
nohup uvicorn web.main:app --host 0.0.0.0 --port 8000 > /tmp/service.log 2>&1 &
```

### 方法3：使用systemd（如果配置了）
```bash
sudo systemctl restart czsc-web
```

## 测试修复效果

1. **重启服务后**，访问：https://www.chanquant.com/analyze/600000

2. **按F12打开开发者工具**

3. **在Console中输入**：
```javascript
console.log(plotData);
console.log(plotData.data);
```

4. **应该看到**：
- `plotData` 是一个对象
- `plotData.data` 是一个数组，包含K线数据

5. **K线图应该正常显示**

## 经验教训

### 1. JSON序列化的层次

- **Python dict** → `json.dumps()` → **JSON string**  
- **JSON string** → `tojson` filter → **Double escaped string** ❌

正确做法：
- **Python dict** → `tojson` filter → **JSON object** ✅

### 2. Jinja2模板最佳实践

当需要将Python对象传递给JavaScript时：

✅ **推荐**:
```javascript
const data = {{ python_dict | tojson | safe }};
```

❌ **不推荐**:
```python
# Python
"json_str": json.dumps(python_dict)
```
```javascript
// JavaScript
const data = JSON.parse('{{ json_str | safe }}');  // 可能双重转义
```

### 3. 调试技巧

当JavaScript对象不符合预期时：

1. 查看生成的HTML源代码
2. 检查JSON是否有多余的引号或转义
3. 在浏览器控制台检查对象类型
4. 使用`typeof`确认数据类型

## Git提交

```
7085c4d - fix: 修复K线图JSON序列化问题
```

## 相关文档

- KLINE_FIX_COMPLETE.md - K线图数据生成
- DATA_DISPLAY_FIX.md - 数据显示问题
- ROUTE_FIX_SUMMARY.md - 路由修复

---

**修复时间**: 2026-01-10 00:55  
**Git哈希**: 7085c4d  
**状态**: 已修复，需要重启服务生效
