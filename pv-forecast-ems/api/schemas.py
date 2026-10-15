"""
API层 - 请求响应数据模型（Pydantic Schema）
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================
# 预测相关
# ============================================

class WeatherForecastPoint(BaseModel):
    """单时刻气象预报数据"""
    timestamp: str = Field(..., description="时间戳 ISO8601")
    ghi: float = Field(..., description="水平面总辐照度 W/㎡", ge=0, le=2000)
    temp: float = Field(..., description="环境温度 ℃", ge=-50, le=60)
    humidity: Optional[float] = Field(None, description="相对湿度 %")
    cloud_cover: Optional[float] = Field(None, description="云量 %")
    wind_speed: Optional[float] = Field(None, description="风速 m/s")


class ForecastRequest(BaseModel):
    """光伏预测请求"""
    weather_forecast: List[WeatherForecastPoint] = Field(..., description="未来气象预报序列")
    model_name: Optional[str] = Field(None, description="指定模型名称，不指定则用最优模型")
    include_physical_constraint: bool = Field(True, description="是否启用物理约束")


class ForecastPoint(BaseModel):
    """单时刻预测结果"""
    timestamp: str
    pv_power_kw: float = Field(..., description="光伏预测功率 kW")
    ghi: Optional[float] = None
    temp: Optional[float] = None
    model_confidence: Optional[float] = Field(None, description="模型置信度 0-1")


class ForecastResponse(BaseModel):
    """光伏预测响应"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any] = Field(..., description="预测结果数据")


# ============================================
# MPC调度相关
# ============================================

class MPCOptimizeRequest(BaseModel):
    """MPC优化请求"""
    pv_forecast_kw: List[float] = Field(..., description="光伏预测功率序列 kW")
    load_kw: Optional[List[float]] = Field(None, description="负荷功率序列 kW")
    price_buy: Optional[List[float]] = Field(None, description="购电电价序列 元/kWh")
    price_sell: Optional[float] = Field(0.3, description="售电电价 元/kWh")
    soc_init: Optional[float] = Field(0.5, description="初始SOC 0-1")
    ess_capacity_kwh: Optional[float] = Field(2000.0, description="储能容量 kWh")
    ess_max_power_kw: Optional[float] = Field(500.0, description="储能最大充放电功率 kW")


class MPCOptimizeResponse(BaseModel):
    """MPC优化响应"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any] = Field(..., description="优化结果")


# ============================================
# 模型管理相关
# ============================================

class ModelInfo(BaseModel):
    """模型信息"""
    model_name: str
    model_type: str
    is_trained: bool
    feature_count: int
    metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class ModelListResponse(BaseModel):
    """模型列表响应"""
    code: int = 200
    message: str = "success"
    data: List[ModelInfo]


class TrainRequest(BaseModel):
    """训练请求"""
    model_types: List[str] = Field(default=["physical", "xgboost", "lstm"],
                                    description="要训练的模型类型")
    data_source: str = Field(default="sample", description="数据源: sample/influxdb")
    lookback_days: int = Field(default=90, description="回溯天数")


class TrainResponse(BaseModel):
    """训练响应"""
    code: int = 200
    message: str = "training_started"
    task_id: str
    data: Dict[str, Any]


# ============================================
# 监控相关
# ============================================

class MonitorMetrics(BaseModel):
    """监控指标"""
    timestamp: str
    nmae_24h: Optional[float] = None
    qualified_rate_24h: Optional[float] = None
    forecast_count_24h: int = 0
    avg_response_time_ms: Optional[float] = None
    model_uptime_hours: float = 0.0


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str
    components: Dict[str, str] = Field(default_factory=dict)
