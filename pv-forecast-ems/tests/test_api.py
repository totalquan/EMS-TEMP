"""
API接口测试
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


class TestHealthCheck:
    """健康检查测试"""

    def test_health_endpoint(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["status"] == "running"


class TestForecastAPI:
    """预测接口测试"""

    def test_pv_forecast_with_valid_data(self):
        """测试正常预测请求"""
        request_data = {
            "weather_forecast": [
                {
                    "timestamp": "2026-09-04T10:00:00+08:00",
                    "ghi": 650.0,
                    "temp": 22.5,
                    "humidity": 45.0,
                    "cloud_cover": 20.0,
                    "wind_speed": 3.5
                },
                {
                    "timestamp": "2026-09-04T10:15:00+08:00",
                    "ghi": 680.0,
                    "temp": 23.0,
                    "humidity": 44.0,
                    "cloud_cover": 15.0,
                    "wind_speed": 3.2
                },
                {
                    "timestamp": "2026-09-04T10:30:00+08:00",
                    "ghi": 700.0,
                    "temp": 23.5,
                    "humidity": 43.0,
                    "cloud_cover": 10.0,
                    "wind_speed": 3.0
                },
                {
                    "timestamp": "2026-09-04T10:45:00+08:00",
                    "ghi": 720.0,
                    "temp": 24.0,
                    "humidity": 42.0,
                    "cloud_cover": 5.0,
                    "wind_speed": 2.8
                }
            ],
            "model_name": "physical_baseline",
            "include_physical_constraint": True
        }

        response = client.post("/api/v1/forecast/pv", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "data" in data
        assert "forecast" in data["data"]
        assert len(data["data"]["forecast"]) == 4

        # 验证预测值在合理范围内
        for item in data["data"]["forecast"]:
            assert 0 <= item["pv_power_kw"] <= 5000
            assert "timestamp" in item

    def test_pv_forecast_night_zero(self):
        """测试夜间预测为0"""
        request_data = {
            "weather_forecast": [
                {
                    "timestamp": "2026-09-04T02:00:00+08:00",
                    "ghi": 0.0,
                    "temp": 15.0
                }
            ],
            "model_name": "physical_baseline"
        }
        response = client.post("/api/v1/forecast/pv", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["forecast"][0]["pv_power_kw"] == 0.0

    def test_pv_forecast_empty_data(self):
        """测试空数据请求"""
        request_data = {"weather_forecast": []}
        response = client.post("/api/v1/forecast/pv", json=request_data)
        assert response.status_code == 400

    def test_mpc_optimize(self):
        """测试MPC优化接口"""
        request_data = {
            "pv_forecast_kw": [1000, 1500, 2000, 2500, 3000, 3200, 3000, 2500,
                                2000, 1500, 1000, 500, 0, 0, 0, 0],
            "load_kw": [2000] * 16,
            "price_buy": [0.5] * 8 + [1.0] * 8,
            "soc_init": 0.5
        }
        response = client.post("/api/v1/forecast/mpc/optimize", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "schedule" in data["data"]
        assert len(data["data"]["schedule"]) == 16
        assert "total_cost_yuan" in data["data"]


class TestModelAPI:
    """模型管理接口测试"""

    def test_list_models(self):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert isinstance(data["data"], list)

    def test_get_physical_model_info(self):
        response = client.get("/api/v1/models/physical_baseline")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["model_type"] == "physical"

    def test_get_active_model(self):
        response = client.get("/api/v1/models/active/current")
        assert response.status_code == 200
        data = response.json()
        assert "active_model" in data["data"]


class TestMonitorAPI:
    """监控接口测试"""

    def test_metrics_endpoint(self):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert "pv_forecast" in response.text

    def test_realtime_monitor(self):
        response = client.get("/api/v1/monitor/realtime")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data["data"]
        assert "system" in data["data"]
