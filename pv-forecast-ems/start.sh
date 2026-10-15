#!/bin/bash
# ============================================
# 光伏预测EMS系统 - 启动脚本
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  光伏预测EMS系统 - 启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装，请先安装Docker${NC}"
    exit 1
fi

# 检查Docker Compose
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}错误: Docker Compose未安装，请先安装Docker Compose${NC}"
    exit 1
fi

# 兼容旧版docker-compose
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}警告: .env文件不存在，正在从.env.example复制...${NC}"
    cp .env.example .env
    echo -e "${GREEN}已创建.env文件，请根据实际情况修改配置${NC}"
fi

# 创建必要目录
mkdir -p logs models/storage models/registry data

# 解析命令行参数
case "${1:-all}" in
    all)
        echo -e "${GREEN}[1/4] 构建镜像...${NC}"
        $DOCKER_COMPOSE build
        echo ""

        echo -e "${GREEN}[2/4] 启动服务...${NC}"
        $DOCKER_COMPOSE up -d
        echo ""

        echo -e "${GREEN}[3/4] 等待服务启动...${NC}"
        sleep 10
        echo ""

        echo -e "${GREEN}[4/4] 服务状态:${NC}"
        $DOCKER_COMPOSE ps
        echo ""

        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  系统启动完成！${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "API文档:    http://localhost:8000/docs"
        echo "健康检查:   http://localhost:8000/api/v1/health"
        echo "Grafana:    http://localhost:3000 (admin/admin123)"
        echo "InfluxDB:   http://localhost:8086"
        echo "MinIO:      http://localhost:9001 (minioadmin/minioadmin)"
        echo ""
        echo "查看日志:   ./start.sh logs"
        echo "停止服务:   ./start.sh stop"
        ;;

    stop)
        echo -e "${YELLOW}停止所有服务...${NC}"
        $DOCKER_COMPOSE down
        echo -e "${GREEN}服务已停止${NC}"
        ;;

    restart)
        echo -e "${YELLOW}重启服务...${NC}"
        $DOCKER_COMPOSE restart
        echo -e "${GREEN}服务已重启${NC}"
        ;;

    logs)
        echo -e "${GREEN}查看日志 (Ctrl+C退出)...${NC}"
        $DOCKER_COMPOSE logs -f --tail=100
        ;;

    status)
        $DOCKER_COMPOSE ps
        ;;

    test)
        echo -e "${GREEN}运行测试...${NC}"
        pip install -r requirements.txt -q
        pytest tests/ -v
        ;;

    train)
        echo -e "${GREEN}训练模型（使用示例数据）...${NC}"
        pip install -r requirements.txt -q
        python -m training.train --models physical xgboost lstm
        ;;

    clean)
        echo -e "${RED}清理所有数据和容器（不可恢复）...${NC}"
        read -p "确定要清理所有数据吗？(yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            $DOCKER_COMPOSE down -v
            rm -rf logs/* models/storage/* models/registry/* data/*
            echo -e "${GREEN}清理完成${NC}"
        else
            echo "已取消"
        fi
        ;;

    *)
        echo "用法: $0 {all|stop|restart|logs|status|test|train|clean}"
        echo ""
        echo "  all      - 构建并启动全部服务（默认）"
        echo "  stop     - 停止所有服务"
        echo "  restart  - 重启所有服务"
        echo "  logs     - 查看实时日志"
        echo "  status   - 查看服务状态"
        echo "  test     - 运行测试用例"
        echo "  train    - 训练模型（示例数据）"
        echo "  clean    - 清理所有数据和容器"
        exit 1
        ;;
esac
