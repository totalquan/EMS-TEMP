"""
模型开发层 - 预测评估指标
业界标准：NMAE（归一化平均绝对误差）、合格率（预测准确率）
而非单纯的MSE/RMSE
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from config.settings import settings


class ForecastEvaluator:
    """光伏预测评估器（业界标准指标）"""

    def __init__(self, capacity_kw: float = None):
        self.capacity = capacity_kw or settings.pv.capacity_kw

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 timestamps: Optional[List] = None) -> Dict[str, Any]:
        """
        综合评估
        :param y_true: 真实值
        :param y_pred: 预测值
        :param timestamps: 时间戳（用于分时段评估）
        :return: 评估指标字典
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # 过滤全零时段（夜间不参与精度评估，避免虚高）
        if timestamps is not None and len(timestamps) == len(y_true):
            daytime_mask = []
            for ts in timestamps:
                hour = ts.hour if hasattr(ts, 'hour') else 12
                daytime_mask.append(6 <= hour <= 18)
            daytime_mask = np.array(daytime_mask)
            if daytime_mask.sum() > 0:
                y_true_day = y_true[daytime_mask]
                y_pred_day = y_pred[daytime_mask]
            else:
                y_true_day = y_true
                y_pred_day = y_pred
        else:
            y_true_day = y_true
            y_pred_day = y_pred

        metrics = {
            # 基础统计
            "sample_count": len(y_true),
            "daytime_sample_count": len(y_true_day),

            # 业界核心指标1：NMAE（归一化平均绝对误差）
            "nmae": self.calculate_nmae(y_true_day, y_pred_day),
            "nmae_all": self.calculate_nmae(y_true, y_pred),

            # 业界核心指标2：合格率（预测准确率，误差<20%装机容量的比例）
            "qualified_rate": self.calculate_qualified_rate(y_true_day, y_pred_day),
            "qualified_rate_all": self.calculate_qualified_rate(y_true, y_pred),

            # 辅助指标
            "mae": self.calculate_mae(y_true_day, y_pred_day),
            "rmse": self.calculate_rmse(y_true_day, y_pred_day),
            "mape": self.calculate_mape(y_true_day, y_pred_day),
            "r2": self.calculate_r2(y_true_day, y_pred_day),
            "bias": self.calculate_bias(y_true_day, y_pred_day),

            # 极值场景评估
            "peak_nmae": self._evaluate_peak(y_true, y_pred, timestamps),
            "ramp_nmae": self._evaluate_ramp(y_true, y_pred),
        }

        # 等级评定
        metrics["grade"] = self._grade(metrics["nmae"], metrics["qualified_rate"])

        logger.info(f"评估完成 | NMAE={metrics['nmae']:.2%} | "
                    f"合格率={metrics['qualified_rate']:.2%} | 等级={metrics['grade']}")
        return metrics

    def calculate_nmae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        NMAE：归一化平均绝对误差（业界最核心指标）
        公式: NMAE = mean(|y_true - y_pred|) / 装机容量
        电网考核标准：日前预测NMAE ≤ 15% 为合格
        """
        if len(y_true) == 0 or self.capacity <= 0:
            return 0.0
        mae = np.mean(np.abs(y_true - y_pred))
        return float(mae / self.capacity)

    def calculate_qualified_rate(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  threshold_ratio: float = 0.2) -> float:
        """
        合格率：预测误差小于阈值（默认20%装机容量）的样本占比
        公式: 合格率 = count(|y_true - y_pred| < 0.2 * 装机) / total
        电网考核标准：合格率 ≥ 80% 为合格
        """
        if len(y_true) == 0:
            return 0.0
        threshold = threshold_ratio * self.capacity
        errors = np.abs(y_true - y_pred)
        qualified = np.sum(errors < threshold)
        return float(qualified / len(y_true))

    def calculate_mae(self, y_true, y_pred):
        """MAE：平均绝对误差"""
        return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0

    def calculate_rmse(self, y_true, y_pred):
        """RMSE：均方根误差"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) > 0 else 0.0

    def calculate_mape(self, y_true, y_pred):
        """MAPE：平均绝对百分比误差（排除零值）"""
        mask = y_true > self.capacity * 0.01
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    def calculate_r2(self, y_true, y_pred):
        """R²：决定系数"""
        if len(y_true) < 2:
            return 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def calculate_bias(self, y_true, y_pred):
        """偏差：平均误差（正=高估，负=低估）"""
        return float(np.mean(y_pred - y_true)) if len(y_true) > 0 else 0.0

    def _evaluate_peak(self, y_true, y_pred, timestamps) -> float:
        """评估峰值时段（出力>80%装机）的NMAE"""
        if timestamps is None:
            return 0.0
        peak_mask = y_true > self.capacity * 0.8
        if peak_mask.sum() == 0:
            return 0.0
        return self.calculate_nmae(y_true[peak_mask], y_pred[peak_mask])

    def _evaluate_ramp(self, y_true, y_pred) -> float:
        """评估爬坡时段（出力变化>30%装机/小时）的NMAE"""
        if len(y_true) < 2:
            return 0.0
        diffs = np.abs(np.diff(y_true))
        ramp_mask = np.append(False, diffs > self.capacity * 0.3)
        if ramp_mask.sum() == 0:
            return 0.0
        return self.calculate_nmae(y_true[ramp_mask], y_pred[ramp_mask])

    def _grade(self, nmae: float, qualified_rate: float) -> str:
        """
        预测精度等级评定（参考电网调度标准）
        S级: NMAE<8% 且 合格率>95%
        A级: NMAE<12% 且 合格率>85%
        B级: NMAE<15% 且 合格率>80%（合格线）
        C级: NMAE<20% 且 合格率>70%
        D级: 不达标
        """
        if nmae < 0.08 and qualified_rate > 0.95:
            return "S（优秀）"
        elif nmae < 0.12 and qualified_rate > 0.85:
            return "A（良好）"
        elif nmae < 0.15 and qualified_rate > 0.80:
            return "B（合格）"
        elif nmae < 0.20 and qualified_rate > 0.70:
            return "C（待改进）"
        else:
            return "D（不合格）"

    def generate_evaluation_report(self, y_true, y_pred, timestamps=None,
                                    model_name: str = "unknown") -> str:
        """生成可读的评估报告"""
        metrics = self.evaluate(y_true, y_pred, timestamps)
        report = f"""
╔══════════════════════════════════════════════════╗
║          光伏预测精度评估报告                      ║
╠══════════════════════════════════════════════════╣
║ 模型名称: {model_name:<38} ║
║ 评估样本: {metrics['sample_count']:<38} ║
║ 白天样本: {metrics['daytime_sample_count']:<38} ║
╠══════════════════════════════════════════════════╣
║ 【核心指标】                                       ║
║ NMAE(归一化平均绝对误差): {metrics['nmae']:>8.2%}          ║
║ 合格率(误差<20%装机占比): {metrics['qualified_rate']:>8.2%}          ║
║ 精度等级: {metrics['grade']:<38} ║
╠══════════════════════════════════════════════════╣
║ 【辅助指标】                                       ║
║ MAE:  {metrics['mae']:>8.2f} kW                            ║
║ RMSE: {metrics['rmse']:>8.2f} kW                            ║
║ MAPE: {metrics['mape']:>8.2%}                              ║
║ R²:   {metrics['r2']:>8.4f}                                ║
║ 偏差: {metrics['bias']:>+8.2f} kW（正=高估/负=低估）       ║
╠══════════════════════════════════════════════════╣
║ 【极值场景】                                       ║
║ 峰值时段NMAE: {metrics['peak_nmae']:>8.2%}                  ║
║ 爬坡时段NMAE: {metrics['ramp_nmae']:>8.2%}                  ║
╚══════════════════════════════════════════════════╝
        """
        return report.strip()
