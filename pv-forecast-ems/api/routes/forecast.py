"""
API层 - 预测服务路由
提供光伏功率预测接口，对接EMS/MPC调度
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from api.schemas import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    MPCOptimizeRequest, MPCOptimizeResponse,
)
from models.physical_model import PhysicalModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from scheduler.mpc_optimizer import MPCOptimizer, MPCOptimizationConfig
from config.settings import settings

router = APIRouter(prefix="/forecast", tags=["光伏预测"])

# 全局模型实例（单例）
_model_instances: Dict[str, Any] = {}
_model_loaded = False


def _load_models():
    """加载已训练的模型"""
    global _model_loaded
    if _model_loaded:
        return

    storage_path = settings.model_storage_path

    # 物理模型始终可用（无需训练）
    physical = PhysicalModel(model_name="physical_baseline")
    physical.train(feature_cols=["ghi", "temp"])
    _model_instances["physical_baseline"] = physical

    # 尝试加载XGBoost
    xgb_path = os.path.join(storage_path, "xgboost_v1.json")
    if os.path.exists(xgb_path):
        xgb = XGBoostModel(model_name="xgboost_v1")
        if xgb.load(xgb_path):
            _model_instances["xgboost_v1"] = xgb

    # 尝试加载LSTM
    lstm_path = os.path.join(storage_path, "lstm_v1.pth")
    if os.path.exists(lstm_path):
        lstm = LSTMModel(model_name="lstm_v1")
        if lstm.load(lstm_path):
            _model_instances["lstm_v1"] = lstm

    _model_loaded = True
    logger.info(f"已加载模型: {list(_model_instances.keys())}")


def _get_best_model_name() -> str:
    """获取最优模型名称（优先XGBoost，其次LSTM，最后物理模型）"""
    priority = ["xgboost_v1", "lstm_v1", "physical_baseline"]
    for name in priority:
        if name in _model_instances:
            return name
    return "physical_baseline"


@router.post("/pv", response_model=ForecastResponse)
async def predict_pv(request: ForecastRequest):
    """
    光伏功率预测接口
    输入未来气象预报，输出光伏功率预测序列
    """
    _load_models()

    try:
        # 1. 解析输入
        weather_data = []
        timestamps = []
        for point in request.weather_forecast:
            weather_data.append({
                "ghi": point.ghi,
                "temp": point.temp,
                "humidity": point.humidity or 50.0,
                "cloud_cover": point.cloud_cover or 0.0,
                "wind_speed": point.wind_speed or 0.0,
            })
            timestamps.append(point.timestamp)

        if not weather_data:
            raise HTTPException(status_code=400, detail="气象预报数据为空")

        # 2. 选择模型
        model_name = request.model_name or _get_best_model_name()
        if model_name not in _model_instances:
            logger.warning(f"模型 {model_name} 未加载，使用默认模型")
            model_name = _get_best_model_name()

        model = _model_instances[model_name]

        # 3. 构造输入特征
        df = pd.DataFrame(weather_data)
        df["timestamp"] = pd.to_datetime(timestamps)

        # 物理模型直接用ghi和temp
        if model.model_type == "physical":
            X = df[["ghi", "temp"]].values
            # 添加太阳高度角和小时
            solar_alt = []
            hours = []
            for ts in df["timestamp"]:
                hour = ts.hour + ts.minute / 60.0
                dayofyear = ts.dayofyear
                declination = 23.45 * np.sin(np.radians(360 * (284 + dayofyear) / 365))
                hour_angle = 15 * (hour - 12)
                sin_alt = (np.sin(np.radians(settings.pv.latitude)) * np.sin(np.radians(declination)) +
                           np.cos(np.radians(settings.pv.latitude)) * np.cos(np.radians(declination)) *
                           np.cos(np.radians(hour_angle)))
                solar_alt.append(np.degrees(np.arcsin(np.clip(sin_alt, -1, 1))))
                hours.append(hour)
            X = np.column_stack([X, solar_alt, hours])
        else:
            # ML模型使用特征工程（简化版）
            from data_pipeline.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()
            df_feat = fe.build_features(df, include_lag=False, include_rolling=False)
            feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "power_kw")]
            if hasattr(model, 'feature_cols') and model.feature_cols:
                feature_cols = [c for c in model.feature_cols if c in df_feat.columns]
            X = df_feat[feature_cols].values

        # 4. 预测
        predictions = model.predict(X)

        # 5. 物理约束
        if request.include_physical_constraint:
            predictions = np.clip(predictions, 0, settings.pv.capacity_kw)
            # 夜间归零
            for i, ts in enumerate(df["timestamp"]):
                if ts.hour < 6 or ts.hour > 18:
                    predictions[i] = 0.0

        # 6. 构造响应
        forecast_points = []
        for i, (ts, pred) in enumerate(zip(timestamps, predictions)):
            forecast_points.append(ForecastPoint(
                timestamp=ts,
                pv_power_kw=round(float(pred), 2),
                ghi=weather_data[i]["ghi"],
                temp=weather_data[i]["temp"],
                model_confidence=0.85,
            ))

        result = {
            "forecast_id": f"PV_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "model_name": model_name,
            "model_type": model.model_type,
            "forecast_horizon": len(forecast_points),
            "time_step_min": settings.forecast.time_step_minutes,
            "generated_at": datetime.now().isoformat(),
            "station": {
                "capacity_kw": settings.pv.capacity_kw,
                "latitude": settings.pv.latitude,
                "longitude": settings.pv.longitude,
                "city": "鄂尔多斯",
            },
            "forecast": [fp.dict() for fp in forecast_points],
        }

        return ForecastResponse(data=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"光伏预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测服务异常: {str(e)}")


@router.post("/mpc/optimize", response_model=MPCOptimizeResponse)
async def optimize_mpc(request: MPCOptimizeRequest):
    """
    MPC联合优化接口
    基于光伏预测，优化储能充放电策略，最小化运行成本
    """
    try:
        # 构造MPC配置
        cfg = MPCOptimizationConfig(
            horizon_steps=len(request.pv_forecast_kw),
            pv_capacity_kw=settings.pv.capacity_kw,
            ess_capacity_kwh=request.ess_capacity_kwh or 2000.0,
            ess_max_power_kw=request.ess_max_power_kw or 500.0,
            price_sell=request.price_sell or 0.3,
        )

        optimizer = MPCOptimizer(cfg)
        result = optimizer.optimize(
            pv_forecast_kw=np.array(request.pv_forecast_kw),
            load_kw=np.array(request.load_kw) if request.load_kw else None,
            price_buy=np.array(request.price_buy) if request.price_buy else None,
            soc_init=request.soc_init,
        )

        return MPCOptimizeResponse(data=result)

    except Exception as e:
        logger.error(f"MPC优化失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MPC优化异常: {str(e)}")


@router.get("/mpc/schedule/{forecast_id}")
async def get_mpc_schedule(forecast_id: str):
    """获取指定预测的MPC调度方案（预留接口）"""
    return MPCOptimizeResponse(data={
        "forecast_id": forecast_id,
        "message": "调度方案查询功能开发中",
    })
