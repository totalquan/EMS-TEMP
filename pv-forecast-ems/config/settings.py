"""
光伏预测EMS工程 - 全局配置管理
统一读取环境变量，提供类型安全的配置访问
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载 .env 文件
load_dotenv()


class PVStationConfig(BaseModel):
    """电站物理参数配置"""
    capacity_kw: float = Field(default=5000.0, description="装机容量kW")
    latitude: float = Field(default=39.61, description="纬度")
    longitude: float = Field(default=110.07, description="经度")
    altitude: float = Field(default=1300.0, description="海拔m")
    panel_tilt: float = Field(default=38.0, description="组件倾角°")
    temp_coefficient: float = Field(default=-0.004, description="功率温度系数/℃")
    system_efficiency: float = Field(default=0.83, description="系统综合效率")
    noct: float = Field(default=45.0, description="组件额定工作温度℃")
    std_ghi: float = Field(default=1000.0, description="标准辐照度W/㎡")
    std_temp: float = Field(default=25.0, description="标准温度℃")


class ForecastConfig(BaseModel):
    """预测参数配置"""
    horizon_hours: int = Field(default=4, description="预测时长小时")
    time_step_minutes: int = Field(default=15, description="时间步长分钟")
    model_type: str = Field(default="lstm", description="模型类型: lstm/transformer/xgboost/physical")
    n_features: int = Field(default=8, description="输入特征数")
    seq_length: int = Field(default=96, description="输入序列长度（15min×96=24h）")


class KafkaConfig(BaseModel):
    """Kafka消息队列配置"""
    bootstrap_servers: str = Field(default="localhost:9092")
    topic_weather: str = Field(default="weather_raw")
    topic_inverter: str = Field(default="inverter_raw")
    topic_meter: str = Field(default="meter_raw")
    group_id: str = Field(default="pv-forecast-group")
    auto_offset_reset: str = Field(default="latest")


class InfluxDBConfig(BaseModel):
    """InfluxDB时序数据库配置"""
    url: str = Field(default="http://localhost:8086")
    token: str = Field(default="")
    org: str = Field(default="ems-org")
    bucket: str = Field(default="pv-forecast")


class S3Config(BaseModel):
    """S3/MinIO对象存储配置（原始数据湖）"""
    endpoint: str = Field(default="http://localhost:9000")
    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")
    bucket: str = Field(default="pv-raw-data")
    region: str = Field(default="us-east-1")


class APIConfig(BaseModel):
    """FastAPI服务配置"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)
    secret_key: str = Field(default="change-me")


class SolverConfig(BaseModel):
    """优化求解器配置（调度层）"""
    engine: str = Field(default="pulp", description="求解引擎: gurobi/pulp/cbc")
    license_path: Optional[str] = Field(default=None, description="Gurobi license路径")
    time_limit: int = Field(default=30, description="求解时间上限秒")


class RetrainConfig(BaseModel):
    """定期重训练配置"""
    enabled: bool = Field(default=True)
    cron: str = Field(default="0 2 * * 0", description="cron表达式")
    lookback_days: int = Field(default=90, description="回溯数据天数")
    min_samples: int = Field(default=1000, description="最小训练样本数")


class Settings(BaseModel):
    """全局配置聚合"""
    # 项目
    project_name: str = Field(default="pv-forecast-ems")
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    base_dir: Path = Field(default=Path(__file__).parent.parent)

    # 各模块配置
    pv: PVStationConfig = Field(default_factory=PVStationConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    s3: S3Config = Field(default_factory=S3Config)
    api: APIConfig = Field(default_factory=APIConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)

    # 路径
    model_storage_path: Path = Field(default=None)
    model_registry_path: Path = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 从环境变量初始化路径
        self.model_storage_path = Path(
            os.getenv("MODEL_STORAGE_PATH", str(self.base_dir / "models" / "storage"))
        )
        self.model_registry_path = Path(
            os.getenv("MODEL_REGISTRY_PATH", str(self.base_dir / "models" / "registry"))
        )
        # 确保目录存在
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.model_registry_path.mkdir(parents=True, exist_ok=True)


def _build_from_env() -> Settings:
    """从环境变量构建设置"""
    return Settings(
        project_name=os.getenv("PROJECT_NAME", "pv-forecast-ems"),
        env=os.getenv("ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        pv=PVStationConfig(
            capacity_kw=float(os.getenv("PV_CAPACITY_KW", 5000)),
            latitude=float(os.getenv("PV_LATITUDE", 39.61)),
            longitude=float(os.getenv("PV_LONGITUDE", 110.07)),
            altitude=float(os.getenv("PV_ALTITUDE", 1300)),
            panel_tilt=float(os.getenv("PV_PANEL_TILT", 38)),
            temp_coefficient=float(os.getenv("PV_TEMP_COEFFICIENT", -0.004)),
            system_efficiency=float(os.getenv("PV_SYSTEM_EFFICIENCY", 0.83)),
        ),
        forecast=ForecastConfig(
            horizon_hours=int(os.getenv("FORECAST_HORIZON_HOURS", 4)),
            time_step_minutes=int(os.getenv("FORECAST_TIME_STEP_MINUTES", 15)),
            model_type=os.getenv("MODEL_TYPE", "lstm"),
        ),
        kafka=KafkaConfig(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic_weather=os.getenv("KAFKA_TOPIC_WEATHER", "weather_raw"),
            topic_inverter=os.getenv("KAFKA_TOPIC_INVERTER", "inverter_raw"),
            topic_meter=os.getenv("KAFKA_TOPIC_METER", "meter_raw"),
            group_id=os.getenv("KAFKA_GROUP_ID", "pv-forecast-group"),
            auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
        ),
        influxdb=InfluxDBConfig(
            url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            token=os.getenv("INFLUXDB_TOKEN", ""),
            org=os.getenv("INFLUXDB_ORG", "ems-org"),
            bucket=os.getenv("INFLUXDB_BUCKET", "pv-forecast"),
        ),
        s3=S3Config(
            endpoint=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
            access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            bucket=os.getenv("S3_BUCKET", "pv-raw-data"),
            region=os.getenv("S3_REGION", "us-east-1"),
        ),
        api=APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", 8000)),
            reload=os.getenv("API_RELOAD", "true").lower() == "true",
            secret_key=os.getenv("API_SECRET_KEY", "change-me"),
        ),
        solver=SolverConfig(
            engine=os.getenv("SOLVER_ENGINE", "pulp"),
            license_path=os.getenv("GUROBI_LICENSE_PATH"),
            time_limit=int(os.getenv("GUROBI_TIME_LIMIT", 30)),
        ),
        retrain=RetrainConfig(
            enabled=os.getenv("RETRAIN_ENABLED", "true").lower() == "true",
            cron=os.getenv("RETRAIN_CRON", "0 2 * * 0"),
            lookback_days=int(os.getenv("RETRAIN_LOOKBACK_DAYS", 90)),
            min_samples=int(os.getenv("RETRAIN_MIN_SAMPLES", 1000)),
        ),
    )


# 全局单例
settings = _build_from_env()
