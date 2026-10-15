"""
模型开发层 - 模型基类
定义所有预测模型的统一接口，支持训练、预测、保存、加载
"""
import os
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


class BasePVModel(ABC):
    """光伏预测模型基类"""

    def __init__(self, model_name: str, model_type: str):
        """
        :param model_name: 模型名称（用于标识和保存）
        :param model_type: 模型类型（lstm/transformer/xgboost/physical）
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_cols: List[str] = []
        self.scaler_params: Dict[str, Tuple[float, float]] = {}
        self.training_info: Dict[str, Any] = {
            "model_name": model_name,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "training_samples": 0,
            "training_time_sec": 0,
            "metrics": {},
        }

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型
        :param X_train: 训练特征
        :param y_train: 训练标签
        :param X_val: 验证特征
        :param y_val: 验证标签
        :return: 训练指标
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        :param X: 输入特征
        :return: 预测值
        """
        pass

    @abstractmethod
    def save(self, save_dir: str) -> str:
        """
        保存模型
        :param save_dir: 保存目录
        :return: 模型文件路径
        """
        pass

    @abstractmethod
    def load(self, model_path: str) -> bool:
        """
        加载模型
        :param model_path: 模型文件路径
        :return: 是否加载成功
        """
        pass

    def predict_with_constraints(self, X: np.ndarray, timestamps: Optional[List[datetime]] = None) -> np.ndarray:
        """
        带物理约束的预测
        :param X: 输入特征
        :param timestamps: 时间戳列表（用于夜间归零）
        :return: 约束后的预测值
        """
        predictions = self.predict(X)

        # 物理约束1：非负
        predictions = np.maximum(predictions, 0.0)

        # 物理约束2：不超过装机容量
        predictions = np.minimum(predictions, settings.pv.capacity_kw)

        # 物理约束3：夜间归零
        if timestamps is not None and len(timestamps) == len(predictions):
            for i, ts in enumerate(timestamps):
                hour = ts.hour if hasattr(ts, 'hour') else 0
                if hour < 6 or hour > 18:
                    predictions[i] = 0.0

        return predictions

    def _save_metadata(self, save_dir: str):
        """保存模型元数据"""
        meta_path = os.path.join(save_dir, f"{self.model_name}_meta.json")
        self.training_info["is_trained"] = self.is_trained
        self.training_info["feature_cols"] = self.feature_cols
        # scaler_params转为可序列化格式
        serializable_scaler = {k: list(v) for k, v in self.scaler_params.items()}
        self.training_info["scaler_params"] = serializable_scaler

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.training_info, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"模型元数据已保存: {meta_path}")

    def _load_metadata(self, save_dir: str) -> bool:
        """加载模型元数据"""
        meta_path = os.path.join(save_dir, f"{self.model_name}_meta.json")
        if not os.path.exists(meta_path):
            logger.warning(f"模型元数据不存在: {meta_path}")
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.training_info.update(meta)
            self.feature_cols = meta.get("feature_cols", [])
            # 恢复scaler_params
            scaler = meta.get("scaler_params", {})
            self.scaler_params = {k: tuple(v) for k, v in scaler.items()}
            self.is_trained = meta.get("is_trained", False)
            logger.info(f"模型元数据已加载: {meta_path}")
            return True
        except Exception as e:
            logger.error(f"加载模型元数据失败: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_cols),
            "training_info": self.training_info,
        }
