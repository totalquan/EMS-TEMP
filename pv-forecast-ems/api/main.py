"""
API层 - FastAPI主入口
光伏预测EMS服务启动文件
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from api.routes.forecast import router as forecast_router
from api.routes.model import router as model_router
from api.routes.monitor import router as monitor_router, record_request


# ============================================
# 日志配置
# ============================================
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.add(
    "logs/pv_forecast_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="INFO",
    encoding="utf-8",
)


# ============================================
# 应用生命周期
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    # 启动时
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║     光伏预测EMS服务启动中...                  ║")
    logger.info(f"║     版本: 1.0.0 | 环境: {settings.env}")
    logger.info(f"║     电站: {settings.pv.capacity_kw/1000:.1f}MW @ 鄂尔多斯")
    logger.info(f"║     预测: {settings.forecast.horizon_hours}h / {settings.forecast.time_step_minutes}min")
    logger.info("╚══════════════════════════════════════════════╝")

    # 确保目录存在
    os.makedirs("logs", exist_ok=True)
    os.makedirs(str(settings.model_storage_path), exist_ok=True)

    yield

    # 关闭时
    logger.info("光伏预测EMS服务已关闭")


# ============================================
# 创建FastAPI应用
# ============================================
app = FastAPI(
    title="光伏预测EMS服务",
    description="""
    基于NWP天气预报的光伏功率预测系统，支持：
    - 多模型预测（物理模型/XGBoost/LSTM/Transformer）
    - MPC联合优化调度（储能充放电策略）
    - 模型管理与自动训练
    - 实时监控与精度评估
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 请求中间件：记录请求时间
# ============================================
@app.middleware("http")
async def add_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    is_forecast = "/forecast/" in request.url.path
    record_request(duration_ms, is_forecast)
    response.headers["X-Process-Time-ms"] = str(round(duration_ms, 2))
    return response


# ============================================
# 全局异常处理
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常 | path={request.url.path} | error={exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "detail": str(exc) if settings.env == "development" else None,
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================
# 注册路由
# ============================================
app.include_router(forecast_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(monitor_router, prefix="/api/v1")


# ============================================
# 根路径
# ============================================
@app.get("/")
async def root():
    return {
        "service": "光伏预测EMS服务",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================
# 启动入口
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level="info",
    )
