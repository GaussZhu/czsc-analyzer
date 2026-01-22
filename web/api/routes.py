"""
主路由模块
聚合所有API端点路由
"""
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
import json
import requests
import logging

# 导入端点路由
from web.api.endpoints import stock, analysis, tracking
from web.api.dependencies import get_analyzer
from app.services.analyzer import StockAnalyzer, get_radar_data

# 配置日志
logger = logging.getLogger(__name__)

# 创建主路由
router = APIRouter()

# 注册子路由
router.include_router(stock.router, prefix="/api/stock", tags=["Stock"])
router.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
router.include_router(tracking.router, prefix="/api/tracking", tags=["Tracking"])

# 配置模板
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ===== 页面路由 =====

@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """首页"""
    return templates.TemplateResponse("pages/index.html", {
        "request": request,
        "title": "CZSC股票分析工具"
    })

@router.get("/analyze/{code}", response_class=HTMLResponse)
async def analyze_stock_page(
    request: Request,
    code: str,
    market: int = 1,
    period: str = "daily",
    analyzer: StockAnalyzer = Depends(get_analyzer)
):
    """
    股票分析页面 - 执行完整分析并渲染结果
    这是原来的设计：用户点击"分析"按钮后跳转到这个页面
    """
    try:
        # 执行分析
        result = analyzer.analyze_stock(
            stock_code=code,
            stock_name=code,  # 暂时使用代码作为名称
            market=market,
            period=period
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="分析失败，请检查股票代码"
            )
        
        # 从result中提取数据
        stock_info = result.get('stock', {})
        signals = result.get('signals', {'buy': {}, 'sell': {}})
        
        # 调试日志
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"分析结果键: {list(result.keys())}")
        logger.info(f"信号数据: buy类型数={len(signals.get('buy', {}))}, sell类型数={len(signals.get('sell', {}))}")
        
        # 获取K线数据用于图表显示
        try:
            kline_data = analyzer.data_source.get_stock_data(
                stock_code=code,
                market=market,
                period=period
            )
            logger.info(f"K线数据获取成功: {len(kline_data) if kline_data is not None else 0} 条")
        except Exception as e:
            logger.error(f"K线数据获取失败: {str(e)}")
            kline_data = None
        
        # 将分类的信号合并成一个列表
        raw_signals = []
        for signal_type, points in signals.get('buy', {}).items():
            logger.info(f"买点类型 {signal_type}: {len(points)} 个")
            for p in points:
                raw_signals.append({
                    'type': f'买点-{signal_type}',
                    'date': str(p.get('time', '')),
                    'price': p.get('price', 0)
                })
        for signal_type, points in signals.get('sell', {}).items():
            logger.info(f"卖点类型 {signal_type}: {len(points)} 个")
            for p in points:
                raw_signals.append({
                    'type': f'卖点-{signal_type}',
                    'date': str(p.get('time', '')),
                    'price': p.get('price', 0)
                })
        
        logger.info(f"合并后的信号总数: {len(raw_signals)}")
        
        # 30分钟追踪
        realtime_tracking = []
        all_buy_points = []
        for signal_type, points in signals.get('buy', {}).items():
            for p in points:
                all_buy_points.append({
                    'type': f'买点-{signal_type}',
                    'time': p.get('time', ''),
                    'price': p.get('price', 0)
                })
        
        if len(all_buy_points) > 0:
            try:
                realtime_tracking = analyzer.get_30min_tracking(code, market, all_buy_points)
                logger.info(f"30分钟追踪: {len(realtime_tracking)} 个买点")
            except Exception as e:
                logger.error(f"30分钟追踪失败: {str(e)}")
                realtime_tracking = []
        
        # 趋势分析
        trend_analysis = None
        try:
            trend_analysis = analyzer.analyze_trend(
                result.get('pivot_points', []),
                kline_data
            )
            if trend_analysis:
                logger.info(f"趋势分析: {trend_analysis.get('trend_type')} - {trend_analysis.get('position')}")
        except Exception as e:
            logger.error(f"趋势分析失败: {str(e)}")
            trend_analysis = None
        
        # 加载AI预测数据
        ai_forecast = {}
        try:
            forecast_file = Path(__file__).parent.parent / "static" / "forecasts" / f"forecast_{code}_B.json"
            if forecast_file.exists():
                with open(forecast_file, 'r', encoding='utf-8') as f:
                    ai_forecast = json.load(f)
                logger.info(f"AI预测数据加载成功: {code}")
            else:
                logger.debug(f"AI预测文件不存在: {forecast_file}")
        except Exception as e:
            logger.error(f"AI预测数据加载失败: {str(e)}")
            ai_forecast = {}
        
        # 构建K线图数据（Plotly Candlestick格式）
        plot_data = {'data': [], 'layout': {}}
        if kline_data is not None and len(kline_data) > 0:
            try:
                # 确保数据按日期排序
                kline_data = kline_data.sort_index()
                
                # 转换日期为字符串格式
                dates = [d.strftime('%Y-%m-%d') for d in kline_data.index]
                
                # 获取最新日期
                latest_date = kline_data.index[-1].strftime('%Y-%m-%d')
                
                # 构建K线蜡烛图数据
                plot_data = {
                    'data': [
                        {
                            'x': dates,
                            'open': kline_data['Open'].tolist(),
                            'high': kline_data['High'].tolist(),
                            'low': kline_data['Low'].tolist(),
                            'close': kline_data['Close'].tolist(),
                            'type': 'candlestick',
                            'name': f'{code} K线',
                            'increasing': {
                                'line': {'color': '#ef5350'}
                            },
                            'decreasing': {
                                'line': {'color': '#26a69a'}
                            }
                        }
                    ],
                    'layout': {
                        'title': {
                            'text': f'{code} 日K线图 (更新至: {latest_date})',
                            'font': {'size': 16}
                        },
                        'xaxis': {
                            'title': '日期',
                            'rangeslider': {'visible': False}
                        },
                        'yaxis': {
                            'title': '价格（元）'
                        },
                        'height': 600,
                        'margin': {'t': 50, 'r': 10, 'l': 60, 'b': 50},
                        'hovermode': 'x unified'
                    }
                }
                logger.info(f"K线图数据构建成功: {len(dates)} 个交易日")
            except Exception as e:
                logger.error(f"K线图数据构建失败: {str(e)}", exc_info=True)
                plot_data = {'data': [], 'layout': {}}
        
        # 准备模板数据
        template_data = {
            "request": request,
            "code": code,
            "title": f"{code} - 股票分析",
            "result": {
                "stock": {
                    "name": stock_info.get('name', code),
                    "code": stock_info.get('code', code)
                },
                "last_price": result.get('last_price', 0),
                "strategy": "缠论分析",
                "fundamental_score": None,  # 暂未实现基本面评分
                "realtime_tracking": realtime_tracking,  # ✅ 已实现30分钟追踪
                "trend_analysis": trend_analysis,  # ✅ 已实现趋势分析
                "fundamental_radar_labels": get_radar_data(code)[0],
                "fundamental_radar_values": get_radar_data(code)[1],
                "raw_signals": raw_signals,
                "ai_forecast": ai_forecast  # ✅ 已实现AI预测加载
            },
            "now": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "is_followed": False,  # 暂未实现关注功能
            "plot_data": plot_data  # 直接传递字典，让模板处理JSON序列化
        }
        
        # 使用完整的analysis模板
        return templates.TemplateResponse("pages/analysis_full.html", template_data)
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"分析失败: {str(e)}", exc_info=True)
        # 返回分析页面但显示错误
        return templates.TemplateResponse("pages/analysis.html", {
            "request": request,
            "code": code,
            "title": "股票分析",
            "error": str(e)
        })

@router.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request, code: str = None):
    """股票查询页面（简化版）"""
    return templates.TemplateResponse("pages/analysis_simple.html", {
        "request": request,
        "title": "股票查询"
    })

@router.get("/tracking", response_class=HTMLResponse)
async def tracking_page_route(request: Request):
    """追踪页面"""
    return templates.TemplateResponse("pages/tracking.html", {
        "request": request,
        "title": "实时追踪"
    })

@router.get("/recommend", response_class=HTMLResponse)
async def recommend_page(request: Request):
    """推荐页面"""
    return templates.TemplateResponse("pages/recommend.html", {
        "request": request,
        "title": "智能推荐"
    })

@router.get("/market", response_class=HTMLResponse)
async def market_overview_page(request: Request):
    """市场概览页面"""
    return templates.TemplateResponse("pages/market_overview.html", {
        "request": request,
        "title": "市场概览"
    })


# ===== API代理端点 =====

@router.get("/api/proxy/tencent")
async def proxy_tencent_api(symbols: str):
    """
    腾讯财经API代理（解决跨域问题）
    
    Args:
        symbols: 股票代码列表，如 sh000001,sz399001
    """
    try:
        url = f"http://qt.gtimg.cn/q={symbols}"
        response = requests.get(url, timeout=10)
        response.encoding = 'gbk'
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="获取数据失败")
        
        return Response(
            content=response.text,
            media_type="text/plain; charset=utf-8"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/ranking/{rank_type}")
async def get_ranking_data(rank_type: str):
    """
    获取排行榜数据（涨幅/跌幅/成交额/换手率）
    
    Args:
        rank_type: 榜单类型 gainers/losers/volume/turnover
    """
    try:
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        # 根据榜单类型设置参数
        if rank_type == 'gainers':
            # 涨幅榜（正序）
            fid = 'f3'  # 按涨跌幅排序
            po = 1      # 正序
        elif rank_type == 'losers':
            # 跌幅榜（倒序）
            fid = 'f3'
            po = 0      # 倒序
        elif rank_type == 'volume':
            # 成交额榜
            fid = 'f6'  # 按成交额排序
            po = 1
        elif rank_type == 'turnover':
            # 换手率榜
            fid = 'f8'  # 按换手率排序
            po = 1
        else:
            raise HTTPException(status_code=400, detail="无效的榜单类型")
        
        params = {
            'pn': 1,           # 页码
            'pz': 50,          # 每页数量（Top50）
            'po': po,          # 排序
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'fid': fid,
            'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',  # 沪深A股
            'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="获取排行榜数据失败")
        
        data = response.json()
        
        # 解析数据
        if data and 'data' in data and 'diff' in data['data']:
            stocks = data['data']['diff']
            
            # 格式化数据
            result = []
            for stock in stocks:
                result.append({
                    'code': stock.get('f12', ''),
                    'name': stock.get('f14', ''),
                    'price': stock.get('f2', 0),
                    'change': stock.get('f3', 0),
                    'volume': round(stock.get('f6', 0) / 100000000, 2),  # 转换为亿
                    'turnover': stock.get('f8', 0)
                })
            
            return {
                'success': True,
                'data': result,
                'count': len(result)
            }
        else:
            raise HTTPException(status_code=500, detail="数据格式异常")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== 高级功能页面 =====

@router.get("/advanced", response_class=HTMLResponse)
async def advanced_page(request: Request):
    """高级功能页面"""
    return templates.TemplateResponse(
        "pages/advanced.html",
        {"request": request}
    )


# ===== 板块行情API =====

@router.get("/api/sectors/{sector_type}")
async def get_sector_data(sector_type: str):
    """
    获取板块行情数据
    
    Args:
        sector_type: 板块类型 concept/industry
    """
    try:
        # 东方财富板块API
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        # 概念板块 m:90+t:3，行业板块 m:90+t:2
        fs = "m:90+t:3" if sector_type == "concept" else "m:90+t:2"
        
        params = {
            'pn': 1,       # 页码
            'pz': 20,      # 每页数量（Top 20）
            'po': 1,       # 正序
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'fid': 'f3',   # 按涨跌幅排序
            'fs': fs,
            'fields': 'f12,f14,f2,f3,f6,f104,f105'  # 代码、名称、涨跌幅、成交额、领涨股、上涨家数、下跌家数
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and data.get('data') and data['data'].get('diff'):
            sectors = data['data']['diff']
            result = []
            
            for sector in sectors:
                result.append({
                    'code': sector.get('f12', ''),
                    'name': sector.get('f14', ''),
                    'change': sector.get('f3', 0),
                    'volume': round(sector.get('f6', 0) / 100000000, 2),  # 转换为亿
                    'leader': sector.get('f104', '-'),
                    'up_count': sector.get('f104', 0),
                    'down_count': sector.get('f105', 0)
                })
            
            return {
                'success': True,
                'data': result,
                'count': len(result)
            }
        else:
            return {'success': False, 'data': [], 'count': 0}
            
    except Exception as e:
        logger.error(f"获取板块数据失败: {e}")
        return {'success': False, 'data': [], 'count': 0}


# ===== 资金流向API =====

@router.get("/api/money-flow/{flow_type}")
async def get_money_flow_data(flow_type: str):
    """
    获取资金流向数据
    
    Args:
        flow_type: 流向类型 inflow/outflow
    """
    try:
        # 东方财富资金流向API
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        params = {
            'pn': 1,
            'pz': 50,      # Top 50
            'po': 1 if flow_type == 'inflow' else 0,  # 净流入正序，净流出倒序
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'fid': 'f62',  # 按主力净额排序
            'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',  # 沪深A股
            'fields': 'f12,f14,f2,f3,f62,f184,f66,f69,f72,f75'  # 代码、名称、价格、涨跌幅、主力净额、超大单、大单、中单、小单
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and data.get('data') and data['data'].get('diff'):
            stocks = data['data']['diff']
            result = []
            
            for stock in stocks:
                result.append({
                    'code': stock.get('f12', ''),
                    'name': stock.get('f14', ''),
                    'price': stock.get('f2', 0),
                    'change': stock.get('f3', 0),
                    'main_money': round(stock.get('f62', 0) / 10000, 2),  # 转换为万
                    'super_large': round(stock.get('f184', 0) / 10000, 2),
                    'large': round(stock.get('f66', 0) / 10000, 2),
                    'medium': round(stock.get('f69', 0) / 10000, 2),
                    'small': round(stock.get('f72', 0) / 10000, 2)
                })
            
            return {
                'success': True,
                'data': result,
                'count': len(result)
            }
        else:
            return {'success': False, 'data': [], 'count': 0}
            
    except Exception as e:
        logger.error(f"获取资金流向失败: {e}")
        return {'success': False, 'data': [], 'count': 0}


# ===== 技术突破选股API =====

@router.get("/api/breakthrough/{signal_type}")
async def get_breakthrough_data(signal_type: str):
    """
    获取技术突破数据
    
    Args:
        signal_type: 信号类型 new_high/volume_breakthrough/limit_up/macd_gold
    """
    try:
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        # 根据信号类型设置参数
        if signal_type == 'new_high':
            # 突破新高（近60日新高）
            fs = 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23'
            fid = 'f3'
            signal_desc = '60日新高'
        elif signal_type == 'volume_breakthrough':
            # 放量突破（换手率>5%且涨幅>3%）
            fs = 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23'
            fid = 'f8'  # 按换手率排序
            signal_desc = '放量突破'
        elif signal_type == 'limit_up':
            # 涨停突破
            fs = 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23'
            fid = 'f3'
            signal_desc = '涨停突破'
        elif signal_type == 'macd_gold':
            # MACD金叉功能已删除（无法准确计算真实MACD指标）
            return {'success': False, 'data': [], 'count': 0, 'message': 'MACD金叉功能已移除'}
        else:
            return {'success': False, 'data': [], 'count': 0}
        
        params = {
            'pn': 1,
            'pz': 50,
            'po': 1,
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'fid': fid,
            'fs': fs,
            'fields': 'f12,f14,f2,f3,f6,f8'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and data.get('data') and data['data'].get('diff'):
            stocks = data['data']['diff']
            result = []
            
            for stock in stocks:
                # 根据信号类型过滤
                change = stock.get('f3', 0)
                turnover = stock.get('f8', 0)
                
                if signal_type == 'volume_breakthrough' and (turnover < 5 or change < 3):
                    continue
                elif signal_type == 'limit_up' and change < 9.5:
                    continue
                
                result.append({
                    'code': stock.get('f12', ''),
                    'name': stock.get('f14', ''),
                    'price': stock.get('f2', 0),
                    'change': stock.get('f3', 0),
                    'volume': round(stock.get('f6', 0) / 100000000, 2),
                    'turnover': stock.get('f8', 0),
                    'signal': signal_desc
                })
            
            return {
                'success': True,
                'data': result[:30],  # 返回前30个
                'count': len(result[:30])
            }
        else:
            return {'success': False, 'data': [], 'count': 0}
            
    except Exception as e:
        logger.error(f"获取技术突破数据失败: {e}")
        return {'success': False, 'data': [], 'count': 0}


# ===== 强势股票池API =====

@router.get("/api/strong-stocks/{stock_type}")
async def get_strong_stocks_data(stock_type: str):
    """
    获取强势股票数据（真实数据源）
    
    Args:
        stock_type: 股票类型 continuous_rise/new_record/strong_consolidation
    """
    try:
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        # 根据类型设置不同的数据源
        if stock_type == 'continuous_rise':
            # 连续涨停（连板股票）- 使用昨日连板板块
            params = {
                'pn': 1,
                'pz': 50,
                'po': 1,
                'np': 1,
                'fltt': 2,
                'invt': 2,
                'fid': 'f3',
                'fs': 'b:BK1051',  # 昨日连板_含一字板块
                'fields': 'f12,f14,f2,f3,f6,f8,f152'  # f152是连板天数字段
            }
            label_field = '连板'
            
        elif stock_type == 'new_record':
            # 强势突破 - 使用阶段新高板块（强势突破股票）
            params = {
                'pn': 1,
                'pz': 50,
                'po': 1,
                'np': 1,
                'fltt': 2,
                'invt': 2,
                'fid': 'f3',
                'fs': 'b:BK0804',  # 阶段新高板块（强势突破，平均涨幅9.57%）
                'fields': 'f12,f14,f2,f3,f6,f8'
            }
            label_field = '强势突破'
            
        else:  # strong_consolidation
            # 强势整理（涨幅3-7%之间的强势股）
            params = {
                'pn': 1,
                'pz': 100,
                'po': 1,
                'np': 1,
                'fltt': 2,
                'invt': 2,
                'fid': 'f3',
                'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',
                'fields': 'f12,f14,f2,f3,f6,f8'
            }
            label_field = '强势'
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and data.get('data') and data['data'].get('diff'):
            stocks = data['data']['diff']
            result = []
            
            for stock in stocks:
                change = stock.get('f3', 0)
                turnover = stock.get('f8', 0)
                
                # 根据类型过滤
                if stock_type == 'continuous_rise':
                    # 连板股票：今日涨幅>0（表示继续上涨）
                    if change <= 0:
                        continue
                    
                    # 使用真实的连板天数字段 f152
                    # f152是昨日连板天数，今天继续涨停则+1
                    lianban_days = stock.get('f152', 0)
                    if lianban_days and lianban_days != '-':
                        # 昨日连板天数 + 今日1天 = 总连板天数
                        total_days = int(lianban_days) + 1
                        days = f"{total_days}连板"
                    else:
                        # 如果没有f152字段，标记为连板但不显示天数
                        days = "连板"
                    
                    period_change = change  # 今日涨幅
                    
                elif stock_type == 'new_record':
                    # 强势突破股票：来自阶段新高板块
                    # 筛选今日大涨的强势突破股票（涨幅>=6%）
                    if change < 6:
                        continue
                    days = "突破"
                    period_change = change
                    
                else:  # strong_consolidation
                    # 强势整理：涨幅在3-7%之间
                    if not (3 <= change < 9):
                        continue
                    days = None
                    period_change = change
                
                result.append({
                    'code': stock.get('f12', ''),
                    'name': stock.get('f14', ''),
                    'price': stock.get('f2', 0),
                    'change': change,
                    'days': days,
                    'period_change': period_change,
                    'volume': round(stock.get('f6', 0) / 100000000, 2)
                })
            
            return {
                'success': True,
                'data': result[:30],  # 返回前30只
                'count': len(result[:30])
            }
        else:
            return {'success': False, 'data': [], 'count': 0}
            
    except Exception as e:
        logger.error(f"获取强势股票失败: {e}")
        return {'success': False, 'data': [], 'count': 0}


# ===== 高级功能概览统计API =====

@router.get("/api/advanced/overview")
async def get_advanced_overview():
    """获取高级功能概览统计数据（真实数据）"""
    try:
        # 获取真实的市场统计数据
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        # 1. 获取所有A股的涨跌统计
        params = {
            'pn': 1,
            'pz': 5000,  # 获取所有股票
            'po': 1,
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'fid': 'f3',
            'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',
            'fields': 'f12,f14,f3,f62'  # f62是主力净流入
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and data.get('data') and data['data'].get('diff'):
            stocks = data['data']['diff']
            
            # 统计数据
            total_inflow = 0  # 主力净流入总额（亿）
            inflow_count = 0  # 净流入家数
            
            for stock in stocks:
                net_flow = stock.get('f62', 0)
                if net_flow and net_flow != '-':
                    try:
                        net_flow_value = float(net_flow)
                        total_inflow += net_flow_value
                        if net_flow_value > 0:
                            inflow_count += 1
                    except:
                        pass
            
            # 转换为亿元（f62单位是元）
            total_inflow_yi = round(total_inflow / 100000000, 1)
            inflow_sign = '+' if total_inflow_yi >= 0 else ''
            
            # 2. 获取板块涨跌统计
            concept_url = "http://push2.eastmoney.com/api/qt/clist/get"
            concept_params = {
                'pn': 1,
                'pz': 200,
                'po': 1,
                'np': 1,
                'fltt': 2,
                'invt': 2,
                'fid': 'f3',
                'fs': 'b:BK0707',  # 概念板块
                'fields': 'f12,f14,f3'
            }
            
            concept_response = requests.get(concept_url, params=concept_params, timeout=10)
            concept_data = concept_response.json()
            
            sector_up = 0
            sector_down = 0
            
            if concept_data and concept_data.get('data') and concept_data['data'].get('diff'):
                for sector in concept_data['data']['diff']:
                    change = sector.get('f3', 0)
                    if change > 0:
                        sector_up += 1
                    elif change < 0:
                        sector_down += 1
            
            return {
                'success': True,
                'data': {
                    'total_inflow': f'{inflow_sign}{total_inflow_yi}亿',
                    'inflow_count': inflow_count,
                    'sector_up': sector_up,
                    'sector_down': sector_down
                }
            }
        else:
            return {'success': False, 'data': {}}
        
    except Exception as e:
        logger.error(f"获取概览统计失败: {e}")
        return {'success': False, 'data': {}}

