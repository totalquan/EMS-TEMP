"""
数据处理层 - 特征工程
构造光伏预测模型所需的输入特征：
- NWP气象特征（辐照度、温度、湿度、云量、风速）
- 历史功率特征（滞后、滑动窗口统计）
- 日历效应特征（小时、月份、季节、是否白天）
- 物理衍生特征（太阳高度角、斜面辐照度估算）
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from config.settings import settings


class FeatureEngineer:
    """光伏预测特征工程"""

    def __init__(self):
        self.lat = settings.pv.latitude
        self.lon = settings.pv.longitude
        self.altitude = settings.pv.altitude
        self.panel_tilt = settings.pv.panel_tilt
        self.pv_capacity = settings.pv.capacity_kw
        self.time_step = settings.forecast.time_step_minutes

        # 特征列定义
        self.weather_features = [
            "ghi", "dhi", "temp", "humidity", "wind_speed",
            "wind_dir", "pressure", "cloud_cover",
        ]
        self.calendar_features = [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "dayofyear_sin", "dayofyear_cos", "is_daytime",
        ]
        self.lag_features = []
        self.rolling_features = []

    def build_features(
        self,
        df: pd.DataFrame,
        include_lag: bool = True,
        include_rolling: bool = True,
        lag_steps: List[int] = None,
        rolling_windows: List[int] = None,
    ) -> pd.DataFrame:
        """
        构建完整特征集
        :param df: 输入数据，需包含timestamp、气象特征、power_kw
        :param include_lag: 是否包含滞后特征
        :param include_rolling: 是否包含滑动窗口特征
        :param lag_steps: 滞后步数列表（15min为单位）
        :param rolling_windows: 滑动窗口大小列表（15min为单位）
        :return: 特征DataFrame
        """
        if df.empty:
            logger.error("输入数据为空，无法构建特征")
            return df

        logger.info(f"开始构建特征 | 输入条数={len(df)}")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 1. 日历效应特征
        df = self._add_calendar_features(df)

        # 2. 物理衍生特征
        df = self._add_physical_features(df)

        # 3. 气象特征处理
        df = self._process_weather_features(df)

        # 4. 历史功率滞后特征
        if include_lag and "power_kw" in df.columns:
            lag_steps = lag_steps or [1, 2, 4, 8, 12, 24, 48, 96]
            df = self._add_lag_features(df, lag_steps)

        # 5. 滑动窗口统计特征
        if include_rolling and "power_kw" in df.columns:
            rolling_windows = rolling_windows or [4, 8, 16, 32, 96]
            df = self._add_rolling_features(df, rolling_windows)

        # 6. 去除因滞后/滑动产生的NaN行
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"去除因特征工程产生的NaN行 {removed} 条")

        logger.info(f"特征构建完成 | 最终条数={len(df)} | 特征列数={len(df.columns) - 2}")
        return df

    def build_forecast_features(
        self,
        weather_forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        为在线预测构建特征（只有未来气象，没有未来功率）
        :param weather_forecast_df: 未来气象预报数据
        :param historical_df: 历史数据（用于计算滞后和滑动特征）
        :return: 预测用特征DataFrame
        """
        if weather_forecast_df.empty:
            logger.error("气象预报数据为空")
            return pd.DataFrame()

        logger.info(f"构建预测特征 | 预报条数={len(weather_forecast_df)} | 历史条数={len(historical_df)}")

        # 合并历史和预报数据（用于计算滞后特征）
        historical_df = historical_df.copy()
        weather_forecast_df = weather_forecast_df.copy()

        # 预报数据中power_kw设为NaN（待预测）
        if "power_kw" not in weather_forecast_df.columns:
            weather_forecast_df["power_kw"] = np.nan

        # 合并并排序
        combined = pd.concat([historical_df, weather_forecast_df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        # 构建特征
        combined = self.build_features(
            combined,
            include_lag=True,
            include_rolling=True,
        )

        # 只取预报时段的特征
        forecast_start = pd.to_datetime(weather_forecast_df["timestamp"].iloc[0])
        forecast_features = combined[combined["timestamp"] >= forecast_start].reset_index(drop=True)

        logger.info(f"预测特征构建完成 | 预报特征条数={len(forecast_features)}")
        return forecast_features

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列名（排除timestamp和目标变量）"""
        exclude = {"timestamp", "power_kw", "power"}
        return [c for c in df.columns if c not in exclude]

    # ============================================
    # 私有方法：各类特征构造
    # ============================================

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加日历效应特征（周期性编码）"""
        ts = df["timestamp"]

        # 小时周期编码（24小时周期）
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

        # 月份周期编码（12月周期）
        df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

        # 年内天数周期编码（365天周期）
        dayofyear = ts.dt.dayofyear
        df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365)
        df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365)

        # 是否白天（6-18点）
        df["is_daytime"] = ((ts.dt.hour >= 6) & (ts.dt.hour <= 18)).astype(int)

        return df

    def _add_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加物理衍生特征"""
        ts = df["timestamp"]

        # 1. 太阳高度角（简化计算，基于纬度和时间）
        dayofyear = ts.dt.dayofyear
        hour = ts.dt.hour + ts.dt.minute / 60.0

        # 太阳赤纬角（近似公式）
        declination = 23.45 * np.sin(np.radians(360 * (284 + dayofyear) / 365))

        # 时角
        hour_angle = 15 * (hour - 12)

        # 太阳高度角
        lat_rad = np.radians(self.lat)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)

        sin_altitude = (
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        )
        df["solar_altitude"] = np.degrees(np.arcsin(np.clip(sin_altitude, -1, 1)))

        # 2. 斜面辐照度估算（简化版，基于水平面辐照度和太阳高度角）
        if "ghi" in df.columns:
            tilt_rad = np.radians(self.panel_tilt)
            alt_rad = np.radians(df["solar_altitude"])
            # 斜面/水平面辐照度比（简化模型）
            tilt_ratio = np.sin(alt_rad + tilt_rad) / np.maximum(np.sin(alt_rad), 0.01)
            tilt_ratio = np.clip(tilt_ratio, 0.5, 2.0)
            df["tilted_ghi_est"] = df["ghi"] * tilt_ratio
            # 夜间为0
            df.loc[df["solar_altitude"] < 0, "tilted_ghi_est"] = 0.0

        # 3. 海拔修正后的大气质量
        df["air_mass"] = 1.0 / np.maximum(np.cos(np.radians(90 - df["solar_altitude"])), 0.05)

        # 4. 温度对功率的影响因子（物理模型预热）
        if "temp" in df.columns:
            df["temp_derating"] = 1 - 0.004 * (df["temp"] - 25)

        return df

    def _process_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理气象特征：归一化、衍生"""
        # 辐照度变化率
        if "ghi" in df.columns:
            df["ghi_diff"] = df["ghi"].diff().fillna(0)
            df["ghi_ratio"] = df["ghi"] / 1000.0  # 归一化到标准辐照度

        # 温度变化率
        if "temp" in df.columns:
            df["temp_diff"] = df["temp"].diff().fillna(0)

        # 风速风向合成（如果有）
        if "wind_speed" in df.columns and "wind_dir" in df.columns:
            df["wind_u"] = df["wind_speed"] * np.cos(np.radians(df["wind_dir"]))
            df["wind_v"] = df["wind_speed"] * np.sin(np.radians(df["wind_dir"]))

        return df

    def _add_lag_features(self, df: pd.DataFrame, lag_steps: List[int]) -> pd.DataFrame:
        """添加历史功率滞后特征"""
        for step in lag_steps:
            col_name = f"power_lag_{step}"
            df[col_name] = df["power_kw"].shift(step)
            self.lag_features.append(col_name)
        return df

    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """添加滑动窗口统计特征"""
        for window in windows:
            # 滑动均值
            mean_col = f"power_roll_mean_{window}"
            df[mean_col] = df["power_kw"].rolling(window=window, min_periods=1).mean()
            self.rolling_features.append(mean_col)

            # 滑动标准差
            std_col = f"power_roll_std_{window}"
            df[std_col] = df["power_kw"].rolling(window=window, min_periods=1).std().fillna(0)
            self.rolling_features.append(std_col)

            # 滑动最大值
            max_col = f"power_roll_max_{window}"
            df[max_col] = df["power_kw"].rolling(window=window, min_periods=1).max()
            self.rolling_features.append(max_col)

        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler_params: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        特征归一化（Min-Max）
        :param df: 特征数据
        :param feature_cols: 要归一化的列
        :param scaler_params: 已有的归一化参数（用于预测时保持一致）
        :return: (归一化后的数据, 归一化参数)
        """
        df = df.copy()

        if scaler_params is None:
            # 训练时计算参数
            scaler_params = {}
            for col in feature_cols:
                if col in df.columns:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    # 避免除零
                    if col_max - col_min < 1e-8:
                        col_max = col_min + 1.0
                    scaler_params[col] = (col_min, col_max)

        # 应用归一化
        for col in feature_cols:
            if col in df.columns and col in scaler_params:
                col_min, col_max = scaler_params[col]
                df[col] = (df[col] - col_min) / (col_max - col_min)
                # 裁剪到[0,1]
                df[col] = df[col].clip(0, 1)

        return df, scaler_params


"""
特征：纬度，经度，海拔，容量
处理特征: 时间方面：小时，天 编码 ； 其他特征：滚动平均






"""
