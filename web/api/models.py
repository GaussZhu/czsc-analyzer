"""
API数据模型
使用Pydantic定义请求和响应模型
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class StockInfo(BaseModel):
    """股票基本信息"""
    code: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    market: int = Field(..., description="市场代码 (0=深圳, 1=上海, 4=北交所)")
    pinyin: Optional[str] = Field(None, description="拼音首字母")

class StockSearchRequest(BaseModel):
    """股票搜索请求"""
    query: str = Field(..., min_length=1, description="搜索关键词（代码/名称/拼音）")

class StockSearchResponse(BaseModel):
    """股票搜索响应"""
    results: List[StockInfo]
    total: int

class AnalysisRequest(BaseModel):
    """分析请求"""
    code: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    market: int = Field(..., description="市场代码")
    period: str = Field(default="daily", description="周期 (daily/30min/weekly)")

class FractalInfo(BaseModel):
    """分型信息"""
    type: str = Field(..., description="分型类型 (顶分型/底分型)")
    date: str = Field(..., description="日期")
    price: float = Field(..., description="价格")
    index: int = Field(..., description="索引位置")

class SignalInfo(BaseModel):
    """买卖点信号"""
    type: str = Field(..., description="信号类型 (买点/卖点)")
    level: int = Field(..., description="级别 (1/2/3)")
    date: str = Field(..., description="日期")
    price: float = Field(..., description="价格")
    description: str = Field(..., description="信号描述")

class DivergenceInfo(BaseModel):
    """背驰信息"""
    type: str = Field(..., description="背驰类型 (看涨/看跌)")
    date: str = Field(..., description="日期")
    price: float = Field(..., description="价格")
    strength: float = Field(..., description="强度")

class AnalysisResult(BaseModel):
    """分析结果"""
    code: str
    name: str
    period: str
    latest_price: float
    latest_date: str
    
    # CZSC分析结果
    fractals: List[FractalInfo]
    signals: List[SignalInfo]
    divergences: List[DivergenceInfo]
    pivots_count: int = Field(0, description="中枢数量")
    strokes_count: int = Field(0, description="笔数量")
    
    # K线数据
    kline_data: Optional[Dict[str, Any]] = None
    
    # 分析时间
    analyzed_at: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    service: str

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
    code: Optional[int] = None
