# 股票分析页面设计说明

## 📋 原始设计理解

### 用户流程
1. 用户在股票列表页面浏览股票
2. 点击某只股票的"分析"按钮
3. 跳转到 `/analyze/{code}` 页面
4. 后端执行完整的CZSC技术分析
5. 返回包含以下内容的完整分析页面：
   - 股票基本信息（代码、名称、当前价格）
   - 交互式K线图（Plotly）
   - 买卖点标注
   - 分型、笔、中枢标注
   - 趋势分析
   - 背驰判断
   - AI预测（可选）
   - 基本面雷达图（可选）

---

## 🔧 技术实现

### 路由设计

#### 1. `/analysis` - 简化入口页面
**用途**：提供输入框，让用户手动输入股票代码

**特点**：
- 前端页面，包含输入框
- JavaScript调用API获取数据
- 适合用户主动查询

**模板**：`web/templates/pages/analysis.html`

#### 2. `/analyze/{code}` - 完整分析页面  
**用途**：执行完整的后端分析并渲染结果

**特点**：
- 后端路由，执行CZSC分析
- 返回包含所有数据的完整页面
- 用户点击股票列表的"分析"按钮后跳转
- 自动执行分析，无需前端API调用

**模板**：`web/templates/pages/analysis_full.html`

**实现代码**：
```python
@router.get("/analyze/{code}", response_class=HTMLResponse)
async def analyze_stock_page(
    request: Request,
    code: str,
    market: int = 1,
    period: str = "daily",
    analyzer: StockAnalyzer = Depends(get_analyzer)
):
    # 执行分析
    result = analyzer.analyze_stock(
        stock_code=code,
        stock_name=code,
        market=market,
        period=period
    )
    
    # 准备模板数据
    template_data = {
        "request": request,
        "result": {
            "stock": {"name": code, "code": code},
            "last_price": result.get('latest_price'),
            "raw_signals": result.get('signals'),
            "trend_analysis": result.get('trend_analysis'),
            ...
        },
        "plot_json": json.dumps(result.get('plot_data'))
    }
    
    return templates.TemplateResponse("pages/analysis_full.html", template_data)
```

---

## 📊 数据流程

### 完整分析流程
```
用户点击"分析" 
    ↓
GET /analyze/600000
    ↓
analyzer.analyze_stock()
    ↓
├─ 获取K线数据（MySQL/TDX）
├─ 计算技术指标（MACD）
├─ 识别分型
├─ 识别笔
├─ 识别中枢
├─ 生成买卖点信号
├─ 判断背驰
└─ 生成Plotly图表数据
    ↓
返回HTML页面（包含所有数据）
    ↓
浏览器渲染K线图和分析结果
```

---

## 🎨 页面组件

### analysis_full.html 包含：

1. **头部信息栏**
   - 股票名称和代码
   - 当前价格
   - 最后更新时间
   - 策略名称
   - 基本面评分（可选）
   - AI预测徽章（可选）

2. **实时追踪区域**（30分钟级别）
   - 来源信号
   - 当前价格
   - 涨幅/回撤
   - 操作建议

3. **基本面雷达图**（可选）
   - 盈利稳定性
   - ROE
   - 负债率
   - 现金流
   - 估值
   - 股息率

4. **交互式K线图**（核心）
   - Plotly.js渲染
   - 包含：
     - K线数据
     - 买卖点标注（三角形）
     - 分型标注
     - 笔的连线
     - 中枢区间
     - AI预测置信带（可选）

5. **趋势分析卡片**
   - 趋势类型（上升/下降/震荡）
   - 当前位置（中枢上沿/下沿/离开）
   - 详细描述
   - 趋势信号
   - 证据说明

6. **操作按钮**
   - 加入关注列表
   - 返回股票列表

---

## 🔑 关键参数

### analyzer.analyze_stock() 参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| stock_code | str | 股票代码 | "600000" |
| stock_name | str | 股票名称 | "浦发银行" |
| market | int | 市场代码 | 1=上海, 0=深圳 |
| period | str | 数据周期 | "daily" 或 "30min" |

### 周期参数映射

| 中文 | 英文 | 说明 |
|------|------|------|
| 日线 | daily | 日K线数据 |
| 30分钟 | 30min | 30分钟K线 |

**⚠️ 注意**：路由参数必须使用英文（"daily"），不要使用中文（"日线"）

---

## 🐛 已知问题与解决

### 1. 参数名错误
**问题**：`analyzer.analyze_stock()` 不接受 `code` 参数  
**解决**：改为 `stock_code` 和 `stock_name`

### 2. 周期参数错误
**问题**：使用中文"日线"导致数据获取失败  
**解决**：改为英文"daily"

### 3. pytdx版本冲突
**现象**：`'bool' object does not support the context manager protocol`  
**影响**：TDX数据源可能不可用  
**解决**：使用MySQL作为后备数据源（已配置自动切换）

### 4. 模板变量缺失
**问题**：原模板期望后端传递复杂的数据结构  
**解决**：
- 简化模板，移除可选功能（AI预测、基本面等）
- 或完善后端，提供完整数据

---

## 📁 相关文件

### 路由文件
- `web/api/routes.py` - 主路由，包含 `/analyze/{code}`

### 模板文件
- `web/templates/pages/analysis.html` - 简化入口页面
- `web/templates/pages/analysis_full.html` - 完整分析页面

### API端点
- `web/api/endpoints/analysis.py` - 分析API实现

### 核心服务
- `app/services/analyzer.py` - StockAnalyzer服务
- `app/core/data_source.py` - 统一数据源

---

## 🎯 当前状态

### ✅ 已完成
- [x] 添加 `/analyze/{code}` 路由
- [x] 恢复 `analysis_full.html` 模板
- [x] 修复API参数名（stock_code）
- [x] 修复周期参数（daily）
- [x] 保留简化的 `/analysis` 作为入口

### ⚠️ 待完善
- [ ] 修复数据获取问题（pytdx版本或使用MySQL）
- [ ] 完善plot_data的生成（K线图数据）
- [ ] 添加基本面评分功能（可选）
- [ ] 添加AI预测功能（可选）
- [ ] 添加股票列表页面，提供"分析"按钮

### 🔧 待测试
- [ ] 完整分析流程测试
- [ ] K线图渲染测试
- [ ] 买卖点标注测试
- [ ] 不同周期数据测试

---

## 💡 使用建议

### 开发环境测试
```bash
# 测试简化入口
curl http://localhost:8000/analysis

# 测试完整分析（需要有数据）
curl http://localhost:8000/analyze/600000
```

### 用户访问
```
# 手动输入代码
https://www.chanquant.com/analysis

# 直接分析（如果有股票列表页面链接）
https://www.chanquant.com/analyze/600000
```

---

## 📝 下一步计划

1. **修复数据源**
   - 检查MySQL数据是否可用
   - 或修复pytdx版本问题
   - 确保至少一个数据源可用

2. **完善图表数据**
   - 确保`plot_data`包含完整的Plotly数据结构
   - 测试K线图渲染

3. **添加股票列表页面**
   - 创建 `/list_stocks` 或 `/stocks` 页面
   - 列出所有股票
   - 每个股票添加"分析"按钮，链接到 `/analyze/{code}`

4. **优化用户体验**
   - 添加加载动画
   - 添加错误提示
   - 添加分享功能

---

*文档更新时间：2026-01-09*  
*项目版本：CZSC v2.1.0*
