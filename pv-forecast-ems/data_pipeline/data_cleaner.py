"""
数据处理层 - 数据清洗与时间戳对齐
支持PySpark（大数据量）和Pandas（中小数据量）双模式
处理：异常值检测、缺失值填充、时间戳对齐、重采样
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger

from config.settings import settings


class DataCleaner:
    """光伏预测数据清洗器"""

    def __init__(self, time_step_minutes: int = None):
        self.time_step = time_step_minutes or settings.forecast.time_step_minutes
        self.pv_capacity = settings.pv.capacity_kw

        # 物理合理范围（用于异常值检测）
        self.valid_ranges = {
            "ghi": (0, 1500),           # 辐照度 W/㎡
            "dhi": (0, 800),             # 散射辐照度
            "temp": (-40, 60),           # 温度 ℃
            "humidity": (0, 100),        # 湿度 %
            "wind_speed": (0, 50),       # 风速 m/s
            "wind_dir": (0, 360),         # 风向 °
            "pressure": (500, 1100),      # 气压 hPa
            "cloud_cover": (0, 100),      # 云量 %
            "power_kw": (0, self.pv_capacity * 1.1),  # 功率 kW（允许10%超配）
            "ac_power": (0, self.pv_capacity * 1000 * 1.1),  # 交流功率 W
        }

    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗气象数据
        :param df: 原始气象DataFrame，需包含timestamp列
        :return: 清洗后的DataFrame
        """
        if df.empty:
            logger.warning("气象数据为空，跳过清洗")
            return df

        logger.info(f"开始清洗气象数据 | 原始条数={len(df)}")
        df = df.copy()

        # 1. 时间戳处理
        df = self._normalize_timestamp(df)

        # 2. 去重
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        # 3. 按时间排序
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 4. 异常值检测与处理
        for col in df.columns:
            if col in self.valid_ranges and col in df.columns:
                low, high = self.valid_ranges[col]
                # 标记异常值为NaN
                mask = (df[col] < low) | (df[col] > high)
                n_outliers = mask.sum()
                if n_outliers > 0:
                    logger.warning(f"列 {col} 检测到 {n_outliers} 个异常值，已标记为缺失")
                    df.loc[mask, col] = np.nan

        # 5. 缺失值处理
        df = self._handle_missing_weather(df)

        # 6. 重采样到固定时间步长
        df = self._resample_to_timestep(df)

        logger.info(f"气象数据清洗完成 | 最终条数={len(df)}")
        return df

    def clean_power_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗光伏功率数据
        :param df: 原始功率DataFrame，需包含timestamp和power_kw列
        :return: 清洗后的DataFrame
        """
        if df.empty:
            logger.warning("功率数据为空，跳过清洗")
            return df

        logger.info(f"开始清洗功率数据 | 原始条数={len(df)}")
        df = df.copy()

        # 1. 时间戳处理
        df = self._normalize_timestamp(df)

        # 2. 去重
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        # 3. 排序
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 4. 物理约束：夜间功率归零
        if "power_kw" in df.columns:
            # 夜间（辐照度接近0时）功率应为0
            night_mask = df["timestamp"].dt.hour.isin(range(0, 6)) | \
                         df["timestamp"].dt.hour.isin(range(19, 24))
            # 允许小的夜间噪声（<1%装机容量）
            noise_threshold = self.pv_capacity * 0.01
            df.loc[night_mask & (df["power_kw"] < noise_threshold), "power_kw"] = 0.0

            # 异常值：超过装机容量110%
            over_cap_mask = df["power_kw"] > self.pv_capacity * 1.1
            n_over = over_cap_mask.sum()
            if n_over > 0:
                logger.warning(f"检测到 {n_over} 个超装机容量异常值，已截断")
                df.loc[over_cap_mask, "power_kw"] = self.pv_capacity

            # 负值处理
            neg_mask = df["power_kw"] < 0
            n_neg = neg_mask.sum()
            if n_neg > 0:
                logger.warning(f"检测到 {n_neg} 个负功率值，已置零")
                df.loc[neg_mask, "power_kw"] = 0.0

        # 5. 缺失值处理（功率数据用插值）
        if "power_kw" in df.columns:
            # method="time" 要求 DatetimeIndex；普通整数索引退化为线性插值
            if isinstance(df.index, pd.DatetimeIndex):
                df["power_kw"] = df["power_kw"].interpolate(method="time", limit=4)
            else:
                df["power_kw"] = df["power_kw"].interpolate(method="linear", limit=4)
            df["power_kw"] = df["power_kw"].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        # 6. 重采样
        df = self._resample_to_timestep(df)

        logger.info(f"功率数据清洗完成 | 最终条数={len(df)}")
        return df

    def align_weather_and_power(
        self, weather_df: pd.DataFrame, power_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        对齐气象数据和功率数据（按时间戳内连接）
        :param weather_df: 清洗后的气象数据
        :param power_df: 清洗后的功率数据
        :return: 对齐后的合并DataFrame
        """
        if weather_df.empty or power_df.empty:
            logger.error("气象或功率数据为空，无法对齐")
            return pd.DataFrame()

        logger.info(f"对齐气象与功率数据 | 气象={len(weather_df)}条 | 功率={len(power_df)}条")

        # 确保时间戳列类型一致
        weather_df = weather_df.copy()
        power_df = power_df.copy()
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])

        # 内连接
        merged = pd.merge(weather_df, power_df, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"数据对齐完成 | 合并后条数={len(merged)}")
        return merged

    def _normalize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化时间戳列"""
        if "timestamp" not in df.columns:
            # 尝试找时间列
            time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_cols:
                df = df.rename(columns={time_cols[0]: "timestamp"})
            else:
                logger.error("数据中未找到时间戳列")
                return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # 转换为本地时间（东八区）
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Shanghai")
        # 去掉时区信息，方便后续处理
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # 去除时间戳为空的行
        n_invalid = df["timestamp"].isna().sum()
        if n_invalid > 0:
            logger.warning(f"去除 {n_invalid} 条无效时间戳数据")
            df = df.dropna(subset=["timestamp"])

        return df

    def _handle_missing_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理气象数据缺失值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "timestamp"]

        for col in numeric_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                # 时间插值（最多填充4个连续缺失）
                df[col] = df[col].interpolate(method="time", limit=4)
                # 剩余缺失用前向/后向填充
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                # 仍有缺失用列均值
                df[col] = df[col].fillna(df[col].mean())

        return df

    def _resample_to_timestep(self, df: pd.DataFrame) -> pd.DataFrame:
        """重采样到固定时间步长"""
        if "timestamp" not in df.columns or df.empty:
            return df

        df = df.set_index("timestamp")
        rule = f"{self.time_step}min"

        # 数值列取均值，非数值列取最后一个
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        resampled = df[numeric_cols].resample(rule).mean()

        # 处理重采样产生的新缺失
        resampled = resampled.interpolate(method="time", limit=2)
        resampled = resampled.reset_index()

        return resampled

    def generate_data_quality_report(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        生成数据质量报告
        :param df: 数据
        :param data_type: 数据类型
        :return: 质量报告字典
        """
        report = {
            "data_type": data_type,
            "total_records": len(df),
            "time_range": None,
            "missing_rate": {},
            "outlier_count": {},
            "duplicate_count": 0,
        }

        if df.empty:
            return report

        # 时间范围
        if "timestamp" in df.columns:
            report["time_range"] = {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            }

        # 缺失率
        for col in df.columns:
            missing_rate = df[col].isna().mean() * 100
            if missing_rate > 0:
                report["missing_rate"][col] = round(missing_rate, 2)

        # 异常值统计
        for col in df.columns:
            if col in self.valid_ranges and col in df.columns:
                low, high = self.valid_ranges[col]
                n_outliers = ((df[col] < low) | (df[col] > high)).sum()
                if n_outliers > 0:
                    report["outlier_count"][col] = int(n_outliers)

        # 重复值
        if "timestamp" in df.columns:
            report["duplicate_count"] = int(df.duplicated(subset=["timestamp"]).sum())

        return report
