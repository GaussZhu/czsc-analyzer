"""
股票分析API端点
整合重构后的CZSC分析服务
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from datetime import datetime

from web.api.models import (
    AnalysisRequest, 
    AnalysisResult, 
    FractalInfo, 
    SignalInfo, 
    DivergenceInfo
)
from web.api.dependencies import get_analyzer, get_data_source
from app.services.analyzer import StockAnalyzer
from app.core.data_source import UnifiedDataSource

router = APIRouter()

def convert_czsc_result_to_response(czsc_result: Dict[str, Any], 
                                    code: str, 
                                    name: str, 
                                    period: str) -> AnalysisResult:
    """
    将CZSC分析结果转换为API响应格式
    """
    # 提取基本信息
    latest_price = czsc_result.get('latest_price', 0.0)
    latest_date = czsc_result.get('latest_date', '')
    if isinstance(latest_date, datetime):
        latest_date = latest_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # 转换分型信息
    fractals = []
    top_fractals = czsc_result.get('fractals', {}).get('top', [])
    bottom_fractals = czsc_result.get('fractals', {}).get('bottom', [])
    
    for f in top_fractals:
        fractals.append(FractalInfo(
            type="顶分型",
            date=f.get('date', ''),
            price=f.get('high', 0.0),
            index=f.get('index', 0)
        ))
    
    for f in bottom_fractals:
        fractals.append(FractalInfo(
            type="底分型",
            date=f.get('date', ''),
            price=f.get('low', 0.0),
            index=f.get('index', 0)
        ))
    
    # 转换买卖点信号
    signals = []
    buy_signals = czsc_result.get('signals', {}).get('buy', [])
    sell_signals = czsc_result.get('signals', {}).get('sell', [])
    
    for s in buy_signals:
        signals.append(SignalInfo(
            type="买点",
            level=s.get('level', 1),
            date=s.get('date', ''),
            price=s.get('price', 0.0),
            description=s.get('description', '')
        ))
    
    for s in sell_signals:
        signals.append(SignalInfo(
            type="卖点",
            level=s.get('level', 1),
            date=s.get('date', ''),
            price=s.get('price', 0.0),
            description=s.get('description', '')
        ))
    
    # 转换背驰信息
    divergences = []
    bullish_div = czsc_result.get('divergences', {}).get('bullish', [])
    bearish_div = czsc_result.get('divergences', {}).get('bearish', [])
    
    for d in bullish_div:
        divergences.append(DivergenceInfo(
            type="看涨背驰",
            date=d.get('date', ''),
            price=d.get('price', 0.0),
            strength=d.get('strength', 0.0)
        ))
    
    for d in bearish_div:
        divergences.append(DivergenceInfo(
            type="看跌背驰",
            date=d.get('date', ''),
            price=d.get('price', 0.0),
            strength=d.get('strength', 0.0)
        ))
    
    return AnalysisResult(
        code=code,
        name=name,
        period=period,
        latest_price=latest_price,
        latest_date=latest_date,
        fractals=fractals,
        signals=signals,
        divergences=divergences,
        pivots_count=czsc_result.get('pivots_count', 0),
        strokes_count=czsc_result.get('strokes_count', 0),
        kline_data=czsc_result.get('kline_data'),
        analyzed_at=datetime.now()
    )

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_stock(
    request: AnalysisRequest,
    analyzer: StockAnalyzer = Depends(get_analyzer)
):
    """
    分析股票
    
    使用CZSC理论进行技术分析，返回：
    - 分型识别结果
    - 笔和中枢
    - 买卖点信号
    - 背驰判断
    """
    try:
        # 执行分析
        result = analyzer.analyze_stock(
            stock_code=request.code,
            stock_name=request.name,
            market=request.market,
            period=request.period
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="分析失败，请检查股票代码和市场参数"
            )
        
        # 转换为API响应格式
        return convert_czsc_result_to_response(
            result, 
            request.code, 
            request.name, 
            request.period
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"分析过程中出错: {str(e)}"
        )

@router.get("/quick/{code}")
async def quick_analyze(
    code: str,
    market: int = 1,
    period: str = "daily",
    analyzer: StockAnalyzer = Depends(get_analyzer)
):
    """
    快速分析（不需要提供完整请求体）
    """
    # 简化的分析，使用默认参数
    result = analyzer.analyze_stock(
        stock_code=code,
        stock_name=code,  # 使用代码作为名称
        market=market,
        period=period
    )
    
    if not result:
        raise HTTPException(
            status_code=500,
            detail="分析失败"
        )
    
    return convert_czsc_result_to_response(
        result,
        code,
        code,
        period
    )

@router.get("/history/{code}")
async def get_analysis_history(
    code: str,
    limit: int = 10
):
    """
    获取历史分析记录
    （暂未实现，预留接口）
    """
    return {
        "code": code,
        "history": [],
        "message": "历史记录功能开发中"
    }

@router.get("/compare")
async def compare_stocks(
    codes: str,
    analyzer: StockAnalyzer = Depends(get_analyzer)
):
    """
    对比多只股票
    codes格式: code1,code2,code3
    """
    code_list = [c.strip() for c in codes.split(',') if c.strip()]
    
    if not code_list:
        raise HTTPException(status_code=400, detail="请提供股票代码")
    
    if len(code_list) > 5:
        raise HTTPException(status_code=400, detail="最多支持对比5只股票")
    
    results = []
    for code in code_list:
        try:
            result = analyzer.analyze_stock(
                stock_code=code,
                stock_name=code,
                market=1,  # 默认上海
                period="daily"
            )
            if result:
                results.append({
                    "code": code,
                    "latest_price": result.get('latest_price', 0),
                    "signals_count": len(result.get('signals', {}).get('buy', [])) + 
                                   len(result.get('signals', {}).get('sell', [])),
                    "fractals_count": len(result.get('fractals', {}).get('top', [])) + 
                                    len(result.get('fractals', {}).get('bottom', []))
                })
        except Exception as e:
            results.append({
                "code": code,
                "error": str(e)
            })
    
    return {
        "codes": code_list,
        "results": results
    }
