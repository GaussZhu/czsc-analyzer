"""
Web应用主入口
FastAPI应用配置和路由注册
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pathlib import Path

# 导入路由
from web.api import routes
from web.core.config import get_web_config

# 创建FastAPI应用
app = FastAPI(
    title="CZSC股票分析工具",
    description="基于缠论的股票技术分析平台",
    version="2.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加缓存控制中间件
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """
    为分析页面添加no-cache头，防止显示旧数据
    """
    response = await call_next(request)
    
    # 对于分析页面，不缓存
    if request.url.path.startswith('/analyze'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response

# 获取配置
config = get_web_config()

# 配置静态文件
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 配置模板
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 注册路由
app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print("🚀 CZSC Web应用启动...")
    print(f"📁 静态文件目录: {STATIC_DIR}")
    print(f"📄 模板目录: {TEMPLATES_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    print("👋 CZSC Web应用关闭")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "service": "CZSC Stock Analyzer"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
