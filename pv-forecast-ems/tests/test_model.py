"""
模型单元测试
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.physical_model import PhysicalModel
from training.evaluate import ForecastEvaluator
from scheduler.mpc_optimizer import MPCOptimizer, MPCOptimizationConfig


class TestPhysicalModel:
    """物理模型测试"""

    def setup_method(self):
        self.model = PhysicalModel(model_name="test_physical")
        self.model.train(feature_cols=["ghi", "temp"])

    def test_daytime_prediction_positive(self):
        """白天预测应为正值"""
        X = np.array([[650.0, 22.5, 45.0, 10]])  # ghi, temp, solar_alt, hour
        pred = self.model.predict(X)
        assert pred[0] > 0
        assert pred[0] <= 5000

    def test_night_prediction_zero(self):
        """夜间预测应为0"""
        X = np.array([[0.0, 15.0, -10.0, 2]])  # 凌晨2点
        pred = self.model.predict(X)
        assert pred[0] == 0.0

    def test_zero_ghi_prediction_zero(self):
        """零辐照度预测应为0"""
        X = np.array([[0.0, 25.0, 30.0, 12]])
        pred = self.model.predict(X)
        assert pred[0] == 0.0

    def test_prediction_within_capacity(self):
        """预测值不超过装机容量"""
        X = np.array([[2000.0, 25.0, 60.0, 12]])  # 超高辐照度
        pred = self.model.predict(X)
        assert pred[0] <= 5000

    def test_batch_prediction(self):
        """批量预测"""
        X = np.array([
            [0.0, 15.0, 0.0, 0],
            [500.0, 20.0, 40.0, 10],
            [800.0, 25.0, 60.0, 12],
            [600.0, 23.0, 50.0, 14],
        ])
        pred = self.model.predict(X)
        assert len(pred) == 4
        assert pred[0] == 0.0  # 夜间
        assert all(p >= 0 for p in pred)


class TestForecastEvaluator:
    """评估指标测试"""

    def setup_method(self):
        self.evaluator = ForecastEvaluator(capacity_kw=5000.0)

    def test_nmae_calculation(self):
        """NMAE计算"""
        y_true = np.array([1000, 2000, 3000, 4000])
        y_pred = np.array([1100, 1900, 3100, 3900])
        nmae = self.evaluator.calculate_nmae(y_true, y_pred)
        expected_mae = np.mean([100, 100, 100, 100])
        expected_nmae = expected_mae / 5000
        assert abs(nmae - expected_nmae) < 1e-6

    def test_qualified_rate(self):
        """合格率计算"""
        y_true = np.array([1000, 2000, 3000])
        y_pred = np.array([1050, 2500, 3050])  # 第二个误差500 > 20%*5000=1000? 不，500<1000，合格
        rate = self.evaluator.calculate_qualified_rate(y_true, y_pred, threshold_ratio=0.2)
        assert rate == 1.0  # 全部合格（误差都<1000）

    def test_grade_assessment(self):
        """等级评定"""
        # S级
        assert "S" in self.evaluator._grade(0.05, 0.98)
        # A级
        assert "A" in self.evaluator._grade(0.10, 0.90)
        # B级（合格）
        assert "B" in self.evaluator._grade(0.14, 0.82)
        # D级（不合格）
        assert "D" in self.evaluator._grade(0.25, 0.60)

    def test_full_evaluation(self):
        """完整评估"""
        y_true = np.random.uniform(0, 4000, 100)
        y_pred = y_true + np.random.normal(0, 200, 100)
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert "nmae" in metrics
        assert "qualified_rate" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "grade" in metrics
        assert 0 <= metrics["nmae"] <= 1
        assert 0 <= metrics["qualified_rate"] <= 1


class TestMPCOptimizer:
    """MPC优化器测试"""

    def setup_method(self):
        self.config = MPCOptimizationConfig(horizon_steps=4, dt_hours=0.25)
        self.optimizer = MPCOptimizer(self.config)

    def test_optimization_returns_schedule(self):
        """优化返回调度方案"""
        pv = np.array([1000, 2000, 3000, 2000])
        load = np.array([2500, 2500, 2500, 2500])
        result = self.optimizer.optimize(pv_forecast_kw=pv, load_kw=load)

        assert result["status"] in ("optimal", "heuristic_fallback")
        assert len(result["schedule"]) == 4
        assert "total_cost_yuan" in result
        assert "final_soc_percent" in result

    def test_power_balance_constraint(self):
        """功率平衡约束验证"""
        pv = np.array([1000, 2000, 3000, 2000])
        load = np.array([2500, 2500, 2500, 2500])
        result = self.optimizer.optimize(pv_forecast_kw=pv, load_kw=load)

        for step in result["schedule"]:
            # 功率平衡：光伏 + 购电 + 放电 = 负荷 + 充电 + 售电
            lhs = step["pv_kw"] + step["grid_buy_kw"] + step["ess_discharge_kw"]
            rhs = step["load_kw"] + step["ess_charge_kw"] + step["grid_sell_kw"]
            assert abs(lhs - rhs) < 1.0

    def test_soc_within_bounds(self):
        """SOC在安全范围内"""
        pv = np.array([1000, 2000, 3000, 2000])
        load = np.array([2500, 2500, 2500, 2500])
        result = self.optimizer.optimize(pv_forecast_kw=pv, load_kw=load, soc_init=0.5)

        for step in result["schedule"]:
            assert 20 <= step["soc_percent"] <= 80

    def test_heuristic_fallback(self):
        """启发式兜底策略"""
        # 强制使用启发式
        self.optimizer.solver_engine = "nonexistent"
        pv = np.array([1000, 2000, 3000, 2000])
        result = self.optimizer.optimize(pv_forecast_kw=pv)
        assert result["status"] == "heuristic_fallback"
        assert len(result["schedule"]) == 4

    def test_padding_short_input(self):
        """短输入自动填充"""
        pv = np.array([1000, 2000])  # 只有2个点，需要填充到16
        result = self.optimizer.optimize(pv_forecast_kw=pv)
        assert len(result["schedule"]) == 16
