"""
优化与选型层 - 模型对比与自动选型
对比多个模型，按NMAE和合格率选择最优模型
"""
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from models.base_model import BasePVModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.xgboost_model import XGBoostModel
from models.physical_model import PhysicalModel
from training.evaluate import ForecastEvaluator
from config.settings import settings


class ModelSelector:
    """模型对比与自动选型器"""

    def __init__(self):
        self.models: Dict[str, BasePVModel] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.evaluator = ForecastEvaluator()

    def register_model(self, model: BasePVModel):
        """注册待对比的模型"""
        self.models[model.model_name] = model
        logger.info(f"注册模型: {model.model_name} ({model.model_type})")

    def register_default_models(self):
        """注册默认对比模型集合"""
        self.register_model(PhysicalModel(model_name="physical_baseline"))
        self.register_model(XGBoostModel(model_name="xgboost_v1"))
        self.register_model(LSTMModel(model_name="lstm_v1", epochs=50))
        self.register_model(TransformerModel(model_name="transformer_v1", epochs=40))
        logger.info(f"已注册 {len(self.models)} 个默认模型")

    def compare_all(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    feature_cols: List[str],
                    test_timestamps: Optional[List] = None) -> Dict[str, Any]:
        """
        训练并对比所有注册模型
        :return: 对比结果
        """
        logger.info(f"开始模型对比 | 模型数={len(self.models)} | 测试样本={len(X_test)}")

        for name, model in self.models.items():
            logger.info(f"━━━ 训练模型: {name} ━━━")
            try:
                start_time = time.time()

                # 训练
                if model.model_type == "physical":
                    model.train(feature_cols=feature_cols)
                else:
                    model.train(X_train, y_train, X_val, y_val, feature_cols=feature_cols)

                # 预测
                if model.model_type in ("lstm", "transformer"):
                    # 序列模型需要特殊处理
                    y_pred = self._predict_sequence_model(model, X_test)
                else:
                    y_pred = model.predict(X_test)

                # 物理约束
                y_pred = np.clip(y_pred, 0, settings.pv.capacity_kw)

                # 评估
                metrics = self.evaluator.evaluate(y_test, y_pred, test_timestamps)

                self.results[name] = {
                    "model_type": model.model_type,
                    "metrics": metrics,
                    "training_time_sec": round(time.time() - start_time, 2),
                    "predictions": y_pred,
                    "model": model,
                }

                logger.info(f"模型 {name} 完成 | NMAE={metrics['nmae']:.2%} | "
                           f"合格率={metrics['qualified_rate']:.2%} | 等级={metrics['grade']}")

            except Exception as e:
                logger.error(f"模型 {name} 训练/评估失败: {e}")
                self.results[name] = {"error": str(e), "metrics": None}

        # 选择最优模型
        self.best_model_name = self._select_best()
        logger.info(f"最优模型: {self.best_model_name}")

        return self.get_comparison_report()

    def _predict_sequence_model(self, model, X_test) -> np.ndarray:
        """序列模型预测（滑动窗口）"""
        seq_length = model.seq_length
        predictions = []
        for i in range(len(X_test)):
            if i < seq_length:
                # 前面的数据不足，用物理模型兜底
                predictions.append(0.0)
            else:
                X_seq = X_test[i - seq_length:i][np.newaxis, :, :]
                pred = model.predict(X_seq)[0]
                predictions.append(pred)
        return np.array(predictions)

    def _select_best(self) -> Optional[str]:
        """
        选择最优模型
        评分规则：NMAE权重60%，合格率权重40%
        """
        best_name = None
        best_score = -float("inf")

        for name, result in self.results.items():
            metrics = result.get("metrics")
            if metrics is None:
                continue
            # 综合评分：NMAE越低越好，合格率越高越好
            score = (1 - metrics["nmae"]) * 0.6 + metrics["qualified_rate"] * 0.4
            result["composite_score"] = round(score, 4)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def get_comparison_report(self) -> Dict[str, Any]:
        """获取对比报告"""
        report = {
            "total_models": len(self.models),
            "successful_models": sum(1 for r in self.results.values() if r.get("metrics") is not None),
            "best_model": self.best_model_name,
            "comparison": [],
        }

        for name, result in self.results.items():
            metrics = result.get("metrics")
            if metrics is None:
                report["comparison"].append({
                    "model_name": name,
                    "model_type": result.get("model_type", "unknown"),
                    "status": "failed",
                    "error": result.get("error", "unknown"),
                })
                continue

            report["comparison"].append({
                "model_name": name,
                "model_type": result.get("model_type", "unknown"),
                "status": "success",
                "nmae": round(metrics["nmae"], 4),
                "qualified_rate": round(metrics["qualified_rate"], 4),
                "mae": round(metrics["mae"], 2),
                "rmse": round(metrics["rmse"], 2),
                "r2": round(metrics["r2"], 4),
                "grade": metrics["grade"],
                "composite_score": result.get("composite_score", 0),
                "training_time_sec": result.get("training_time_sec", 0),
            })

        # 按综合评分排序
        report["comparison"].sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return report

    def save_comparison_report(self, save_path: str):
        """保存对比报告为JSON"""
        report = self.get_comparison_report()
        # 移除predictions和model对象（不可序列化）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"模型对比报告已保存: {save_path}")

    def get_best_model(self) -> Optional[BasePVModel]:
        """获取最优模型实例"""
        if self.best_model_name is None:
            return None
        return self.results[self.best_model_name].get("model")
