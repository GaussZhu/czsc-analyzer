"""
股票追踪API端点
用于实时追踪和推荐功能
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import json
from pathlib import Path

router = APIRouter()

# 追踪和推荐数据路径
TRACKING_DATA_PATH = Path(__file__).parent.parent.parent / "static" / "tracking_history.json"
RECOMMEND_DATA_PATH = Path(__file__).parent.parent.parent / "static" / "recommend_result.json"

@router.get("/list")
async def get_tracking_list():
    """
    获取追踪列表
    """
    try:
        if TRACKING_DATA_PATH.exists():
            content = TRACKING_DATA_PATH.read_text(encoding='utf-8')
            data = json.loads(content)
            return {
                "success": True,
                "data": data,
                "total": len(data) if isinstance(data, list) else 0
            }
        
        return {
            "success": False,
            "data": [],
            "total": 0,
            "message": "追踪数据未找到"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取追踪数据失败: {str(e)}"
        )

@router.get("/recommend")
async def get_recommend_list():
    """
    获取推荐列表
    """
    try:
        if RECOMMEND_DATA_PATH.exists():
            content = RECOMMEND_DATA_PATH.read_text(encoding='utf-8')
            data = json.loads(content)
            return {
                "success": True,
                "data": data,
                "total": len(data) if isinstance(data, list) else 0
            }
        
        return {
            "success": False,
            "data": [],
            "total": 0,
            "message": "推荐数据未找到"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取推荐数据失败: {str(e)}"
        )

@router.post("/add")
async def add_to_tracking(code: str, name: str, market: int):
    """
    添加到追踪列表
    （功能开发中）
    """
    return {
        "success": True,
        "message": f"已添加 {name}({code}) 到追踪列表",
        "code": code,
        "name": name,
        "market": market
    }

@router.delete("/remove/{code}")
async def remove_from_tracking(code: str):
    """
    从追踪列表移除
    （功能开发中）
    """
    return {
        "success": True,
        "message": f"已从追踪列表移除 {code}",
        "code": code
    }

@router.get("/stats")
async def get_tracking_stats():
    """
    获取追踪统计信息
    """
    try:
        tracking_count = 0
        recommend_count = 0
        
        if TRACKING_DATA_PATH.exists():
            content = TRACKING_DATA_PATH.read_text(encoding='utf-8')
            data = json.loads(content)
            tracking_count = len(data) if isinstance(data, list) else 0
        
        if RECOMMEND_DATA_PATH.exists():
            content = RECOMMEND_DATA_PATH.read_text(encoding='utf-8')
            data = json.loads(content)
            recommend_count = len(data) if isinstance(data, list) else 0
        
        return {
            "tracking_count": tracking_count,
            "recommend_count": recommend_count,
            "total": tracking_count + recommend_count
        }
    except Exception as e:
        return {
            "tracking_count": 0,
            "recommend_count": 0,
            "total": 0,
            "error": str(e)
        }
