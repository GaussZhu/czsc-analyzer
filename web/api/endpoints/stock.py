"""
股票搜索和信息API端点
"""
from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List
import pandas as pd
from pypinyin import lazy_pinyin
from pytdx.hq import TdxHq_API

from web.api.models import StockInfo, StockSearchResponse
from web.api.dependencies import get_app_config

router = APIRouter()

# 缓存股票列表
_stock_list_cache: List[StockInfo] = []

def load_stock_list() -> List[StockInfo]:
    """加载股票列表"""
    global _stock_list_cache
    
    if _stock_list_cache:
        return _stock_list_cache
    
    try:
        # 尝试从CSV加载
        import os
        csv_path = "/root/stock_monitor_project/all_stock_list.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            _stock_list_cache = [
                StockInfo(
                    code=row['code'],
                    name=row['name'],
                    market=int(row['market']),
                    pinyin=row.get('pinyin', '')
                )
                for _, row in df.iterrows()
            ]
            return _stock_list_cache
    except Exception as e:
        print(f"从CSV加载股票列表失败: {e}")
    
    # 如果CSV不存在，从TDX获取
    return fetch_stock_list_from_tdx()

def fetch_stock_list_from_tdx() -> List[StockInfo]:
    """从TDX获取股票列表"""
    global _stock_list_cache
    
    try:
        api = TdxHq_API()
        api.connect('119.147.212.81', 7709)
        
        stock_list = []
        for market in [0, 1, 4]:  # 深圳、上海、北交所
            for i in range(10):
                stocks = api.get_security_list(market, i * 1000)
                if not stocks:
                    break
                    
                for stock in stocks:
                    name = stock['name']
                    initials = ''.join([p[0].upper() for p in lazy_pinyin(name)])
                    stock_list.append(StockInfo(
                        code=stock['code'],
                        name=name,
                        market=market,
                        pinyin=initials
                    ))
        
        api.disconnect()
        _stock_list_cache = stock_list
        
        # 保存到CSV
        df = pd.DataFrame([s.dict() for s in stock_list])
        df.to_csv("/root/stock_monitor_project/all_stock_list.csv", 
                  index=False, encoding="utf-8-sig")
        
        return stock_list
    except Exception as e:
        print(f"从TDX获取股票列表失败: {e}")
        return []

@router.get("/search", response_model=StockSearchResponse)
async def search_stock(
    q: str = Query(..., min_length=1, description="搜索关键词（代码/名称/拼音）")
):
    """
    搜索股票
    
    支持：
    - 股票代码（如: 600000）
    - 股票名称（如: 浦发银行）
    - 拼音首字母（如: PFYH）
    """
    stock_list = load_stock_list()
    
    if not stock_list:
        raise HTTPException(status_code=503, detail="股票列表未加载")
    
    q = q.strip().upper()
    results = []
    
    for stock in stock_list:
        # 匹配代码
        if q in stock.code:
            results.append(stock)
            continue
        
        # 匹配名称
        if q in stock.name:
            results.append(stock)
            continue
        
        # 匹配拼音
        if stock.pinyin and q in stock.pinyin:
            results.append(stock)
    
    # 限制返回数量
    results = results[:50]
    
    return StockSearchResponse(
        results=results,
        total=len(results)
    )

@router.get("/list", response_model=StockSearchResponse)
async def list_stocks(
    market: int = Query(None, description="市场代码 (0=深圳, 1=上海, 4=北交所)"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量")
):
    """
    获取股票列表
    """
    stock_list = load_stock_list()
    
    if not stock_list:
        raise HTTPException(status_code=503, detail="股票列表未加载")
    
    # 按市场筛选
    if market is not None:
        stock_list = [s for s in stock_list if s.market == market]
    
    # 限制数量
    stock_list = stock_list[:limit]
    
    return StockSearchResponse(
        results=stock_list,
        total=len(stock_list)
    )

@router.post("/refresh")
async def refresh_stock_list():
    """
    刷新股票列表
    """
    global _stock_list_cache
    _stock_list_cache = []
    
    stock_list = fetch_stock_list_from_tdx()
    
    return {
        "message": "股票列表刷新成功",
        "total": len(stock_list)
    }

@router.get("/info/{code}")
async def get_stock_info(code: str):
    """
    获取股票详细信息
    """
    stock_list = load_stock_list()
    
    for stock in stock_list:
        if stock.code == code:
            return stock
    
    raise HTTPException(status_code=404, detail=f"未找到股票: {code}")
