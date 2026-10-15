"""
模型开发层 - XGBoost基线模型
工业界最常用的传统ML基线，稳定、可解释、训练快
"""
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from models.base_model import BasePVModel


class XGBoostModel(BasePVModel):
    """XGBoost光伏预测模型（工业基线）"""

    def __init__(self, model_name: str = "xgboost_pv_default",
                 n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: int = 5,
                 reg_alpha: float = 0.1, reg_lambda: float = 1.0):
        super().__init__(model_name=model_name, model_type="xgboost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if xgb is None:
            raise RuntimeError("xgboost 未安装，请执行 pip install xgboost")

        start_time = time.time()
        logger.info(f"开始训练XGBoost模型 | 训练样本={len(X_train)}")
        self.feature_cols = kwargs.get("feature_cols", [])

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols) if X_val is not None else None

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": "hist",
            "verbosity": 0,
        }

        evals = [(dtrain, "train")]
        if dval:
            evals.append((dval, "val"))

        self.model = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
            evals=evals, early_stopping_rounds=30, verbose_eval=False
        )

        self.is_trained = True
        best_iter = self.model.best_iteration if hasattr(self.model, "best_iteration") else self.n_estimators

        self.training_info.update({
            "training_samples": len(X_train),
            "training_time_sec": round(time.time() - start_time, 2),
            "best_iteration": best_iter,
            "feature_importance": self.get_feature_importance(),
            "hyperparameters": params,
        })
        logger.info(f"XGBoost训练完成 | best_iter={best_iter}")
        return self.training_info

    def predict(self, X):
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型未训练")
        dtest = xgb.DMatrix(X, feature_names=self.feature_cols)
        return self.model.predict(dtest)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.model is None:
            return {}
        importance = self.model.get_score(importance_type="gain")
        total = sum(importance.values())
        return {k: round(v / total, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}.json")
        if self.model:
            self.model.save_model(path)
        self._save_metadata(save_dir)
        logger.info(f"XGBoost模型已保存: {path}")
        return path

    def load(self, model_path):
        if xgb is None or not os.path.exists(model_path):
            return False
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.is_trained = True
            self._load_metadata(os.path.dirname(model_path))
            return True
        except Exception as e:
            logger.error(f"加载XGBoost失败: {e}")
            return False
