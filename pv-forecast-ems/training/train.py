"""
训练入口 - 端到端训练流程
数据加载 → 清洗 → 特征工程 → 模型训练 → 评估 → 保存
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from data_pipeline.data_cleaner import DataCleaner
from data_pipeline.feature_engineer import FeatureEngineer
from models.model_selector import ModelSelector
from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel
from models.physical_model import PhysicalModel
from training.evaluate import ForecastEvaluator


class TrainingPipeline:
    """端到端训练流水线"""

    def __init__(self):
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ForecastEvaluator()
        self.selector = ModelSelector()
        self.scaler_params = None

    def load_sample_data(self, n_days: int = 30) -> pd.DataFrame:
        """
        加载示例数据（实际使用时从InfluxDB读取）
        生成模拟的光伏+气象数据用于演示
        """
        logger.info(f"生成示例数据 | 天数={n_days}")
        np.random.seed(42)

        dates = pd.date_range(end=datetime.now(), periods=n_days * 96, freq="15min")
        n = len(dates)

        # 气象特征（基于时间的周期性 + 噪声）
        hour = dates.hour + dates.minute / 60.0
        dayofyear = dates.dayofyear

        # 辐照度：日出日落曲线 + 季节变化 + 噪声
        solar_elevation = np.maximum(0, np.sin(np.radians(15 * (hour - 12) + 90 * np.sin(np.radians(360 * (dayofyear - 80) / 365)))))
        ghi = solar_elevation * 1000 * (0.8 + 0.2 * np.random.randn(n))
        ghi = np.maximum(0, ghi)

        temp = 15 + 10 * np.sin(np.radians(15 * (hour - 14))) + 15 * np.sin(np.radians(360 * (dayofyear - 200) / 365)) + np.random.randn(n) * 2
        humidity = 50 + 20 * np.sin(np.radians(15 * (hour - 6))) + np.random.randn(n) * 10
        cloud_cover = np.random.uniform(0, 100, n)
        wind_speed = np.random.uniform(1, 8, n)

        # 光伏功率：基于物理公式 + 噪声
        temp_factor = 1 - 0.004 * (temp - 25)
        power = (ghi / 1000) * settings.pv.capacity_kw * temp_factor * settings.pv.system_efficiency
        power = power * (0.95 + 0.1 * np.random.randn(n))  # 加5-10%噪声
        power = np.clip(power, 0, settings.pv.capacity_kw)
        # 夜间归零
        power[(hour < 6) | (hour > 18)] = 0

        df = pd.DataFrame({
            "timestamp": dates,
            "ghi": ghi,
            "temp": temp,
            "humidity": humidity,
            "cloud_cover": cloud_cover,
            "wind_speed": wind_speed,
            "power_kw": power,
        })

        logger.info(f"示例数据生成完成 | 条数={len(df)} | 时间范围={df['timestamp'].min()} ~ {df['timestamp'].max()}")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """数据预处理：清洗 → 特征工程 → 数据集划分"""
        logger.info("开始数据预处理...")

        # 1. 清洗
        df = self.cleaner.clean_weather_data(df)
        df = self.cleaner.clean_power_data(df)

        # 2. 特征工程
        df = self.feature_engineer.build_features(df, include_lag=True, include_rolling=True)

        # 3. 获取特征列
        feature_cols = self.feature_engineer.get_feature_columns(df)
        logger.info(f"特征列数: {len(feature_cols)}")

        # 4. 归一化
        X = df[feature_cols].values
        y = df["power_kw"].values
        X, self.scaler_params = self.feature_engineer.normalize_features(
            pd.DataFrame(X, columns=feature_cols), feature_cols
        )
        X = X.values

        # 5. 数据集划分（时间序列：70%训练，15%验证，15%测试）
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        test_timestamps = df["timestamp"].iloc[val_end:].tolist()

        logger.info(f"数据集划分 | 训练={len(X_train)} | 验证={len(X_val)} | 测试={len(X_test)}")
        return (X_train, y_train, X_val, y_val, X_test, y_test,
                feature_cols, test_timestamps)

    def run(self, df: Optional[pd.DataFrame] = None,
            model_types: Optional[list] = None,
            save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整训练流程
        :param df: 输入数据，None则生成示例数据
        :param model_types: 要训练的模型类型列表
        :param save_dir: 模型保存目录
        """
        logger.info("╔══════════════════════════════════════╗")
        logger.info("║     光伏预测模型训练流水线启动        ║")
        logger.info("╚══════════════════════════════════════╝")

        # 1. 加载数据
        if df is None:
            df = self.load_sample_data(n_days=30)

        # 2. 数据准备
        (X_train, y_train, X_val, y_val, X_test, y_test,
         feature_cols, test_timestamps) = self.prepare_data(df)

        # 3. 注册模型
        if model_types is None:
            model_types = ["physical", "xgboost", "lstm"]

        for mt in model_types:
            if mt == "physical":
                self.selector.register_model(PhysicalModel(model_name="physical_baseline"))
            elif mt == "xgboost":
                self.selector.register_model(XGBoostModel(model_name="xgboost_v1"))
            elif mt == "lstm":
                self.selector.register_model(LSTMModel(model_name="lstm_v1", epochs=30))
            elif mt == "transformer":
                from models.transformer_model import TransformerModel
                self.selector.register_model(TransformerModel(model_name="transformer_v1", epochs=20))

        # 4. 训练对比
        report = self.selector.compare_all(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, test_timestamps
        )

        # 5. 保存最优模型
        if save_dir is None:
            save_dir = str(settings.model_storage_path)

        best_model = self.selector.get_best_model()
        if best_model:
            model_path = best_model.save(save_dir)
            logger.info(f"最优模型已保存: {model_path}")

            # 保存scaler参数
            scaler_path = os.path.join(save_dir, f"{best_model.model_name}_scaler.json")
            with open(scaler_path, "w") as f:
                json.dump({k: list(v) for k, v in self.scaler_params.items()}, f, indent=2)

        # 6. 保存对比报告
        report_path = os.path.join(save_dir, "model_comparison_report.json")
        self.selector.save_comparison_report(report_path)

        logger.info("╔══════════════════════════════════════╗")
        logger.info(f"║  训练完成 | 最优模型: {self.selector.best_model_name}  ║")
        logger.info("╚══════════════════════════════════════╝")

        return report


def main():
    parser = argparse.ArgumentParser(description="光伏预测模型训练")
    parser.add_argument("--models", nargs="+", default=["physical", "xgboost", "lstm"],
                       help="要训练的模型类型")
    parser.add_argument("--save-dir", type=str, default=None, help="模型保存目录")
    args = parser.parse_args()

    pipeline = TrainingPipeline()
    report = pipeline.run(model_types=args.models, save_dir=args.save_dir)

    print("\n" + "=" * 60)
    print("模型对比结果:")
    for item in report["comparison"]:
        if item["status"] == "success":
            print(f"  {item['model_name']:<25} NMAE={item['nmae']:.2%}  "
                  f"合格率={item['qualified_rate']:.2%}  等级={item['grade']}")
    print(f"\n最优模型: {report['best_model']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
