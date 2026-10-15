# 光伏预测EMS系统 - 工程化完整实现

> 基于NWP天气预报的分布式光伏功率预测与能量管理系统，覆盖数据接入、特征工程、模型训练、预测服务、MPC调度优化、容器化部署、监控运维全流程。

## 项目特性

- **多模型支持**：物理模型（新建园区冷启动）、XGBoost、LSTM、Transformer
- **业界标准评估**：NMAE（归一化平均绝对误差）、合格率（电网考核标准），非单纯MSE
- **MPC联合优化**：基于预测结果优化储能充放电策略，最小化运行成本（支持Gurobi/PuLP双引擎）
- **完整数据链路**：Kafka实时接入 → InfluxDB时序存储 → MinIO/S3原始数据湖
- **工程化封装**：FastAPI服务、Docker容器化、K8s编排、Grafana监控
- **自动迭代**：定期重训练、模型自动选型切换、精度实时监控

## 技术栈

| 层级 | 技术选型 |
|------|----------|
| 数据接入 | Apache Kafka |
| 时序存储 | InfluxDB 2.x |
| 数据湖 | MinIO (S3兼容) |
| 数据处理 | Pandas / PySpark |
| 模型框架 | PyTorch (LSTM/Transformer) / XGBoost |
| 优化求解 | Gurobi / PuLP+CBC |
| API服务 | FastAPI + Uvicorn |
| 容器化 | Docker + Docker Compose |
| 编排 | Kubernetes + HPA |
| 监控 | Prometheus + Grafana |
| 定时任务 | APScheduler |

## 快速开始

### 1. 环境要求
- Python 3.11+
- Docker & Docker Compose（推荐）
- 至少4GB内存，10GB磁盘空间

### 2. 一键启动（Docker Compose）

```bash
# 克隆项目
cd pv-forecast-ems

# 复制环境变量配置
cp .env.example .env

# 启动全部服务（预测服务 + InfluxDB + Kafka + MinIO + Grafana + Prometheus）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f pv-forecast
```

### 3. 本地开发运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 训练模型（使用示例数据）
python -m training.train --models physical xgboost lstm

# 运行测试
pytest tests/ -v
```

## API接口

服务启动后，访问 `http://localhost:8000/docs` 查看完整Swagger文档。

### 核心接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/forecast/pv` | POST | 光伏功率预测 |
| `/api/v1/forecast/mpc/optimize` | POST | MPC调度优化 |
| `/api/v1/models` | GET | 模型列表 |
| `/api/v1/models/train` | POST | 触发模型训练 |
| `/api/v1/models/{name}/activate` | POST | 切换激活模型 |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/metrics` | GET | Prometheus指标 |
| `/api/v1/monitor/realtime` | GET | 实时监控数据 |

### 光伏预测请求示例

```bash
curl -X POST http://localhost:8000/api/v1/forecast/pv \
  -H "Content-Type: application/json" \
  -d '{
    "weather_forecast": [
      {"timestamp": "2026-09-04T10:00:00+08:00", "ghi": 650.0, "temp": 22.5},
      {"timestamp": "2026-09-04T10:15:00+08:00", "ghi": 680.0, "temp": 23.0}
    ],
    "model_name": "physical_baseline"
  }'
```

## 项目结构

```
pv-forecast-ems/
├── api/                          # FastAPI服务层
│   ├── main.py                   # 应用入口
│   ├── schemas.py                # 请求响应模型
│   └── routes/
│       ├── forecast.py           # 预测接口
│       ├── model.py              # 模型管理接口
│       └── monitor.py            # 监控接口
├── config/
│   └── settings.py               # 全局配置管理
├── data_pipeline/                # 数据处理层
│   ├── kafka_consumer.py         # Kafka数据接入
│   ├── influxdb_writer.py        # InfluxDB读写
│   ├── s3_storage.py             # S3/MinIO数据湖
│   ├── data_cleaner.py           # 数据清洗与对齐
│   └── feature_engineer.py       # 特征工程
├── models/                       # 模型层
│   ├── base_model.py             # 模型基类
│   ├── physical_model.py         # 物理模型（新建园区冷启动）
│   ├── xgboost_model.py          # XGBoost模型
│   ├── lstm_model.py             # LSTM模型
│   ├── transformer_model.py      # Transformer模型
│   └── model_selector.py         # 模型对比与选型
├── training/                     # 训练与评估
│   ├── train.py                  # 训练流水线入口
│   ├── evaluate.py               # 评估指标（NMAE/合格率）
│   └── retrain_scheduler.py      # 定期重训练调度
├── scheduler/                    # 调度优化层
│   └── mpc_optimizer.py          # MPC模型预测控制优化
├── deployment/                   # 部署配置
│   ├── docker/                   # Docker相关
│   ├── k8s/                      # Kubernetes编排
│   ├── prometheus/               # Prometheus配置
│   └── grafana/                  # Grafana看板
├── tests/                        # 测试
│   ├── test_api.py
│   └── test_model.py
├── docs/                         # 文档
│   ├── ARCHITECTURE.md           # 架构说明
│   ├── DEPLOYMENT.md             # 部署指南
│   └── CHECKLIST.md              # 上线检查清单
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 模型说明

### 1. 物理模型（新建园区首选）
- 基于辐照度-温度-效率公式，无需历史数据训练
- 公式：`P = (GHI/1000) × 装机容量 × 温度修正 × 系统效率`
- 适用于新建园区冷启动阶段

### 2. XGBoost（工业基线）
- 梯度提升树，稳定可解释，训练快
- 适用于有1-3个月历史数据的场景

### 3. LSTM（时序深度学习）
- 长短期记忆网络，捕捉时序依赖
- 适用于有6个月以上历史数据的场景

### 4. Transformer（先进模型）
- 自注意力机制，捕捉长程依赖
- 适用于数据充足、追求极致精度的场景

## 评估指标（电网考核标准）

| 指标 | 公式 | 合格标准 |
|------|------|----------|
| NMAE | mean(\|y_true - y_pred\|) / 装机容量 | ≤ 15% |
| 合格率 | count(\|误差\| < 20%装机) / 总样本 | ≥ 80% |
| 等级 | S/A/B/C/D | B级及以上合格 |

## MPC调度优化

基于光伏预测结果，优化储能充放电策略：
- 目标：最小化（购电成本 - 售电收入 + 储能损耗成本）
- 约束：功率平衡、SOC上下限、电网功率上下限
- 求解器：Gurobi（商业）/ PuLP+CBC（开源，默认）

## 监控与运维

- **Grafana看板**：预测精度、服务状态、系统资源实时监控
- **Prometheus指标**：请求量、响应时间、NMAE、合格率
- **定期重训练**：每周自动重训练，自动切换最优模型
- **告警**：精度不达标、服务异常自动告警

## 文档

- [架构说明](docs/ARCHITECTURE.md)
- [部署指南](docs/DEPLOYMENT.md)
- [上线检查清单](docs/CHECKLIST.md)

## License

MIT
