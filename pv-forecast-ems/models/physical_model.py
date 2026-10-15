"""
模型开发层 - 纯物理预测模型
适用于新建园区无历史数据场景，基于辐照度-温度-效率物理公式
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger

from models.base_model import BasePVModel
from config.settings import settings


class PhysicalModel(BasePVModel):
    """纯物理光伏预测模型（新建园区冷启动用）"""

    def __init__(self, model_name: str = "physical_pv_default"):
        super().__init__(model_name=model_name, model_type="physical")
        self.capacity = settings.pv.capacity_kw
        self.std_ghi = settings.pv.std_ghi
        self.std_temp = settings.pv.std_temp
        self.temp_coefficient = settings.pv.temp_coefficient
        self.system_efficiency = settings.pv.system_efficiency
        self.noct = settings.pv.noct
        self.panel_tilt = settings.pv.panel_tilt
        self.altitude = settings.pv.altitude
        self.is_trained = True  # 物理模型无需训练
        self.training_info["note"] = "纯物理模型，无需训练，基于辐照度-温度-效率公式"

    def train(self, X_train=None, y_train=None, X_val=None, y_val=None, **kwargs):
        """物理模型无需训练，直接返回"""
        logger.info("物理模型无需训练，已就绪")
        self.feature_cols = kwargs.get("feature_cols", ["ghi", "temp"])
        self.training_info["feature_cols"] = self.feature_cols
        return self.training_info

    def _calculate_tilted_ghi(self, ghi: float, solar_altitude: float = None) -> float:
        """计算斜面辐照度（简化模型）"""
        if solar_altitude is None or solar_altitude <= 0:
            # 无太阳高度角时用经验系数
            tilt_factor = 1.0 + np.cos(np.radians(self.panel_tilt)) / 2
        else:
            alt_rad = np.radians(solar_altitude)
            tilt_rad = np.radians(self.panel_tilt)
            tilt_factor = np.sin(alt_rad + tilt_rad) / max(np.sin(alt_rad), 0.01)
            tilt_factor = np.clip(tilt_factor, 0.5, 2.0)

        # 海拔修正
        altitude_factor = 1 + 0.0001 * self.altitude
        return ghi * tilt_factor * altitude_factor

    def _calculate_cell_temp(self, ambient_temp: float, tilted_ghi: float) -> float:
        """计算电池工作温度"""
        return ambient_temp + (tilted_ghi / 800.0) * (self.noct - 20.0)

    def _calculate_single(self, ghi: float, temp: float,
                          solar_altitude: float = None, hour: int = None) -> float:
        """计算单时刻功率"""
        # 夜间归零
        if hour is not None and (hour < 6 or hour > 18):
            return 0.0
        if ghi <= 10:
            return 0.0

        # 1. 斜面辐照度
        tilted_ghi = self._calculate_tilted_ghi(ghi, solar_altitude)

        # 2. 电池温度
        cell_temp = self._calculate_cell_temp(temp, tilted_ghi)

        # 3. 温度修正
        temp_factor = 1 + self.temp_coefficient * (cell_temp - self.std_temp)

        # 4. 理论功率
        theoretical = (tilted_ghi / self.std_ghi) * self.capacity * temp_factor

        # 5. 系统效率
        actual = theoretical * self.system_efficiency

        # 6. 物理约束
        return max(0.0, min(self.capacity, actual))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        X列顺序需包含: ghi, temp（可选: solar_altitude, hour）
        """
        predictions = []
        for row in X:
            ghi = float(row[0]) if len(row) > 0 else 0.0
            temp = float(row[1]) if len(row) > 1 else 25.0
            solar_alt = float(row[2]) if len(row) > 2 else None
            hour = int(row[3]) if len(row) > 3 else None
            predictions.append(self._calculate_single(ghi, temp, solar_alt, hour))
        return np.array(predictions)

    def save(self, save_dir: str) -> str:
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}_params.json")
        params = {
            "capacity": self.capacity, "std_ghi": self.std_ghi,
            "std_temp": self.std_temp, "temp_coefficient": self.temp_coefficient,
            "system_efficiency": self.system_efficiency, "noct": self.noct,
            "panel_tilt": self.panel_tilt, "altitude": self.altitude,
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        self._save_metadata(save_dir)
        return path

    def load(self, model_path: str) -> bool:
        import os, json
        if not os.path.exists(model_path):
            return False
        try:
            with open(model_path, "r") as f:
                params = json.load(f)
            self.capacity = params.get("capacity", self.capacity)
            self.system_efficiency = params.get("system_efficiency", self.system_efficiency)
            self.is_trained = True
            self._load_metadata(os.path.dirname(model_path))
            return True
        except Exception as e:
            logger.error(f"加载物理模型失败: {e}")
            return False
