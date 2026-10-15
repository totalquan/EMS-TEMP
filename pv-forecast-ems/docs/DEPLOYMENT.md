# 部署指南

## 一、环境要求

### 硬件要求
| 环境 | CPU | 内存 | 磁盘 | 说明 |
|------|-----|------|------|------|
| 开发/测试 | 4核 | 8GB | 50GB | 单机Docker Compose |
| 生产（最小） | 8核 | 16GB | 200GB | 预测服务+数据库 |
| 生产（推荐） | 16核 | 32GB | 500GB | 全组件高可用 |

### 软件要求
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+（K8s部署时）
- Python 3.11+（本地开发时）

## 二、Docker Compose 部署（推荐快速上手）

### 1. 准备配置

```bash
# 进入项目目录
cd pv-forecast-ems

# 复制环境变量模板
cp .env.example .env

# 编辑配置（根据实际情况修改）
vim .env
```

关键配置项：
```env
# 电站参数（鄂尔多斯5MW示例）
PV_CAPACITY_KW=5000
PV_LATITUDE=39.61
PV_LONGITUDE=110.07
PV_ALTITUDE=1300

# 预测参数
FORECAST_HORIZON_HOURS=4
FORECAST_TIME_STEP_MINUTES=15

# InfluxDB（如使用外部InfluxDB，修改此处）
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=pv-forecast-token

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# 求解器（pulp=开源免费，gurobi=商业需license）
SOLVER_ENGINE=pulp
```

### 2. 启动服务

```bash
# 构建并启动全部服务（后台运行）
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 预期输出：
# NAME                 STATUS
# pv-forecast-service  running (healthy)
# pv-influxdb          running
# pv-kafka             running
# pv-minio             running
# pv-grafana           running
# pv-prometheus        running
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 预期返回：
# {"status":"healthy","version":"1.0.0",...}

# 查看API文档
# 浏览器打开 http://localhost:8000/docs

# 查看Grafana看板
# 浏览器打开 http://localhost:3000
# 默认账号：admin / admin123
```

### 4. 测试预测接口

```bash
# 测试光伏预测（物理模型）
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

### 5. 常用运维命令

```bash
# 查看日志
docker-compose logs -f pv-forecast          # 预测服务日志
docker-compose logs -f influxdb              # InfluxDB日志
docker-compose logs -f kafka                 # Kafka日志

# 重启服务
docker-compose restart pv-forecast

# 停止服务
docker-compose down

# 停止并清除数据（谨慎！）
docker-compose down -v
```

## 三、Kubernetes 部署（生产环境）

### 1. 准备镜像

```bash
# 构建镜像
docker build -t registry.example.com/ems/pv-forecast:v1.0.0 .

# 推送到镜像仓库
docker push registry.example.com/ems/pv-forecast:v1.0.0
```

### 2. 创建命名空间

```bash
kubectl create namespace ems-system
```

### 3. 部署基础组件（InfluxDB/Kafka/MinIO）

可使用官方Helm Chart部署：
```bash
# InfluxDB
helm repo add influxdata https://helm.influxdata.com/
helm install influxdb influxdata/influxdb2 -n ems-system

# Kafka
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install kafka bitnami/kafka -n ems-system

# MinIO
helm install minio bitnami/minio -n ems-system
```

### 4. 部署预测服务

```bash
# 修改 deployment/k8s/deployment.yaml 中的镜像地址
# image: registry.example.com/ems/pv-forecast:v1.0.0

# 部署
kubectl apply -f deployment/k8s/deployment.yaml -n ems-system

# 查看部署状态
kubectl get pods -n ems-system -l app=pv-forecast
kubectl get svc -n ems-system pv-forecast-service
kubectl get hpa -n ems-system
```

### 5. 配置Ingress（可选）

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pv-forecast-ingress
  namespace: ems-system
spec:
  rules:
    - host: pv-forecast.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: pv-forecast-service
                port:
                  number: 8000
```

## 四、本地开发部署

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动依赖服务（仅启动数据库等）

```bash
# 只启动基础组件，不启动预测服务
docker-compose up -d influxdb kafka minio
```

### 3. 启动开发服务

```bash
# 启动FastAPI（热重载）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行
python api/main.py
```

### 4. 训练模型

```bash
# 使用示例数据训练
python -m training.train --models physical xgboost lstm

# 指定保存目录
python -m training.train --models xgboost lstm --save-dir ./models/storage
```

### 5. 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行API测试
pytest tests/test_api.py -v

# 运行模型测试
pytest tests/test_model.py -v
```

## 五、Gurobi求解器配置（可选）

默认使用开源的PuLP+CBC求解器，如需使用Gurobi（商业求解器，速度更快）：

### 1. 获取Gurobi License
- 学术用户：申请免费学术license
- 商业用户：购买商业license

### 2. 配置环境变量
```env
SOLVER_ENGINE=gurobi
GUROBI_LICENSE_PATH=/opt/gurobi/gurobi.lic
GUROBI_TIME_LIMIT=30
```

### 3. Docker中使用Gurobi
在Dockerfile中添加Gurobi安装，并挂载license文件：
```yaml
volumes:
  - /path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro
```

## 六、数据迁移与备份

### InfluxDB备份
```bash
# 备份
docker exec pv-influxdb influx backup /backup/$(date +%Y%m%d)

# 恢复
docker exec pv-influxdb influx restore /backup/20260904
```

### 模型文件备份
模型文件存储在 `models/storage/` 目录，定期备份到S3或其他存储。

## 七、常见问题

### Q1: 服务启动后健康检查失败
- 检查InfluxDB/Kafka是否正常启动
- 查看日志：`docker-compose logs pv-forecast`
- 确认环境变量配置正确

### Q2: 预测结果全为0
- 检查输入的时间戳是否为夜间（小时<6或>18）
- 检查GHI辐照度是否为0
- 物理模型夜间输出为0是正常的

### Q3: MPC优化返回heuristic_fallback
- 默认使用PuLP求解器，如求解失败会回退到启发式策略
- 可尝试安装Gurobi提升求解稳定性
- 检查输入数据是否合理（如预测时长是否匹配）

### Q4: Kafka连接失败
- 确认Kafka服务正常启动：`docker-compose ps kafka`
- 检查Kafka端口是否被占用
- 确认`KAFKA_BOOTSTRAP_SERVERS`配置正确

### Q5: InfluxDB写入失败
- 确认InfluxDB token正确
- 检查bucket是否存在
- 查看InfluxDB日志：`docker-compose logs influxdb`
