"""
训练层 - 滚动窗口（Walk-Forward）交叉验证
时序数据严禁随机切分：随机切分会让模型"看到未来"，线下指标虚高、上线就崩。
本模块按时间轴滚动：用过去训练 → 预测下一段 → 窗口前移 → 重复，模拟真实上线后的表现。

支持两种模式：
- expanding：训练窗口不断扩大（用全部历史，适合数据少的起步期）
- sliding：训练窗口固定长度前移（只用最近 N 天，适合有明显分布漂移的场景，如季节/积灰）

用法：
    python training/rolling_validation.py --days 60 --models physical xgboost
    python training/rolling_validation.py --mode sliding --train-days 30
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from data_pipeline.data_cleaner import DataCleaner
from data_pipeline.feature_engineer import FeatureEngineer
from training.evaluate import ForecastEvaluator


class RollingValidator:
    """滚动窗口交叉验证器"""

    def __init__(self,
                 initial_train_days: int = 14,
                 test_window_days: int = 7,
                 step_days: int = 7,
                 mode: str = "expanding",
                 max_folds: Optional[int] = None):
        """
        :param initial_train_days: 第一折的训练窗口天数
        :param test_window_days: 每折的测试窗口天数（模拟"未来一周"）
        :param step_days: 窗口前移步长（天）
        :param mode: expanding（扩张窗）/ sliding（滑动窗）
        :param max_folds: 最多跑几折（None=按数据自动算）
        """
        assert mode in ("expanding", "sliding"), f"未知模式: {mode}"
        self.initial_train_days = initial_train_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.mode = mode
        self.max_folds = max_folds
        self.evaluator = ForecastEvaluator()

    # ------------------------------------------------------------------
    # 切分：只按时间索引，绝不随机
    # ------------------------------------------------------------------
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        按时间滚动切分。
        :return: [(train_df, test_df, val_df), ...]  val 从 train 尾部切 20%（不占测试窗）
        """
        t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
        total_days = (t_max - t_min).total_seconds() / 86400

        folds = []
        cursor = t_min + pd.Timedelta(days=self.initial_train_days)

        while cursor + pd.Timedelta(days=self.test_window_days) <= t_max:
            if self.max_folds and len(folds) >= self.max_folds:
                break

            train_start = t_min if self.mode == "expanding" else cursor - pd.Timedelta(days=self.initial_train_days)

            train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < cursor)].copy()
            test_df = df[(df["timestamp"] >= cursor) &
                         (df["timestamp"] < cursor + pd.Timedelta(days=self.test_window_days))].copy()

            if len(train_df) >= 96 * 2 and len(test_df) >= 96:  # 至少2天训练+1天测试
                folds.append((train_df, test_df))

            cursor += pd.Timedelta(days=self.step_days)

        logger.info(f"滚动切分完成 | 总跨度={total_days:.1f}天 | 折数={len(folds)} | 模式={self.mode}")
        for i, (tr, te) in enumerate(folds):
            logger.info(f"  Fold {i}: train {tr['timestamp'].min()} ~ {tr['timestamp'].max()} "
                        f"({len(tr)}条) | test {te['timestamp'].min()} ~ {te['timestamp'].max()} ({len(te)}条)")
        return folds

    # ------------------------------------------------------------------
    # 单折执行：scaler 只在训练窗拟合（防泄漏的关键）
    # ------------------------------------------------------------------
    def _run_fold(self,
                  train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  model_factory: Callable,
                  feature_cols: List[str]) -> Dict[str, Any]:
        fe = FeatureEngineer()

        # 1. 归一化参数只从训练窗学
        X_train_raw = train_df[feature_cols].values
        y_train = train_df["power_kw"].values
        X_train, scaler_params = fe.normalize_features(
            pd.DataFrame(X_train_raw, columns=feature_cols), feature_cols
        )
        X_train = X_train.values

        # 2. 用同一套参数变换测试窗（不重新拟合！）
        X_test, _ = fe.normalize_features(
            pd.DataFrame(test_df[feature_cols].values, columns=feature_cols),
            feature_cols, scaler_params=scaler_params
        )
        X_test = X_test.values
        y_test = test_df["power_kw"].values

        # 3. 从验证尾切 20%（仍不碰测试窗）
        val_end = int(len(X_train) * 0.8)
        X_tr, y_tr = X_train[:val_end], y_train[:val_end]
        X_val, y_val = X_train[val_end:], y_train[val_end:]

        # 4. 训练 + 预测
        model = model_factory()
        if model.model_type == "physical":
            model.train(feature_cols=feature_cols)
        else:
            model.train(X_tr, y_tr, X_val, y_val, feature_cols=feature_cols)
        y_pred = model.predict(X_test)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, settings.pv.capacity_kw)

        # 5. 电网口径评估
        metrics = self.evaluator.evaluate(
            y_test, y_pred, test_df["timestamp"].tolist()
        )
        return {"model": model.model_name, **metrics}

    # ------------------------------------------------------------------
    # 完整验证：逐折跑 + 汇总均值/波动 + 漂移趋势
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame,
                 model_factories: Dict[str, Callable],
                 feature_cols: List[str]) -> Dict[str, Any]:
        """
        :param model_factories: {名称: 工厂函数} 工厂每次返回全新模型实例（折间不共享状态）
        :return: 验证总报告
        """
        folds = self.split(df)
        if not folds:
            logger.error("数据跨度不足以切出任何一折，请增大 --days 或减小 initial_train_days")
            return {"status": "insufficient_data"}

        all_results: Dict[str, List[Dict]] = {name: [] for name in model_factories}

        for i, (train_df, test_df) in enumerate(folds):
            logger.info(f"━━━ Fold {i + 1}/{len(folds)} ━━━")
            for name, factory in model_factories.items():
                try:
                    fold_result = self._run_fold(train_df, test_df, factory, feature_cols)
                    fold_result["fold"] = i
                    fold_result["test_start"] = str(test_df["timestamp"].min())
                    all_results[name].append(fold_result)
                    logger.info(f"  {name:<18} NMAE={fold_result['nmae']:.2%} "
                                f"合格率={fold_result['qualified_rate']:.2%} 等级={fold_result['grade']}")
                except Exception as e:
                    logger.error(f"  {name} 第{i}折失败: {e}")
                    all_results[name].append({"fold": i, "status": "failed", "error": str(e)})

        # 汇总
        summary = {}
        for name, folds_results in all_results.items():
            ok = [r for r in folds_results if r.get("status") != "failed"]
            if not ok:
                summary[name] = {"status": "all_failed"}
                continue
            nmaes = [r["nmae"] for r in ok]
            summary[name] = {
                "folds_completed": len(ok),
                "nmae_mean": float(np.mean(nmaes)),
                "nmae_std": float(np.std(nmaes)),
                "nmae_worst": float(np.max(nmaes)),
                "qualified_rate_mean": float(np.mean([r["qualified_rate"] for r in ok])),
                "grade": self.evaluator._grade(np.mean(nmaes), np.mean([r["qualified_rate"] for r in ok])),
                # 漂移信号：NMAE 随折数的线性趋势，斜率>0 说明越近越差（分布漂移/性能衰退）
                "nmae_trend_per_fold": self._trend_slope(nmaes),
                "drift_suspected": self._trend_slope(nmaes) > 0.002,
                "fold_details": folds_results,
            }
            logger.info(
                f"【汇总】{name}: NMAE均值={np.mean(nmaes):.2%} ± {np.std(nmaes):.2%} "
                f"最差={np.max(nmaes):.2%} 等级={summary[name]['grade']} "
                f"漂移嫌疑={'是' if summary[name]['drift_suspected'] else '否'}"
            )

        # 稳定性排名：不看均值看"最差折"——上线看的是你哪天最烂
        ranked = sorted(
            [(n, s) for n, s in summary.items() if s.get("status") != "all_failed"],
            key=lambda x: x[1]["nmae_worst"]
        )
        report = {
            "mode": self.mode,
            "initial_train_days": self.initial_train_days,
            "test_window_days": self.test_window_days,
            "step_days": self.step_days,
            "n_folds": len(folds),
            "summary": summary,
            "best_by_worst_fold": ranked[0][0] if ranked else None,
            "generated_at": datetime.now().isoformat(),
        }
        return report

    @staticmethod
    def _trend_slope(values: List[float]) -> float:
        """NMAE随折数变化的最小二乘斜率（每折）"""
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 示例数据（自包含，不依赖 torch/LSTM 导入链；接InfluxDB后替换此函数）
# ----------------------------------------------------------------------
def _gen_sample_data(n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.now().floor("15min"), periods=n_days * 96, freq="15min")
    n = len(dates)
    hour = dates.hour + dates.minute / 60.0
    dayofyear = dates.dayofyear

    solar_elev = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    season = 0.7 + 0.3 * np.sin(np.radians(360 * (dayofyear - 80) / 365))
    ghi = solar_elev * season * 1000 * (0.8 + 0.2 * rng.standard_normal(n))
    ghi = np.maximum(ghi, 0)

    temp = 15 + 10 * np.sin(np.radians(15 * (hour - 14))) + \
        15 * np.sin(np.radians(360 * (dayofyear - 200) / 365)) + rng.standard_normal(n) * 2
    cloud = rng.uniform(0, 100, n)

    cap = settings.pv.capacity_kw
    power = (ghi / 1000) * cap * (1 - 0.004 * (temp - 25)) * settings.pv.system_efficiency
    power = power * (0.95 + 0.1 * rng.standard_normal(n)) * (1 - cloud / 400).clip(0.5, 1)
    power = np.clip(power, 0, cap)
    power = np.where((hour < 6) | (hour > 19), 0, power)

    return pd.DataFrame({
        "timestamp": dates,
        "ghi": ghi, "temp": temp, "humidity": rng.uniform(20, 90, n),
        "cloud_cover": cloud, "wind_speed": rng.uniform(1, 8, n),
        "power_kw": power,
    })


def main():
    parser = argparse.ArgumentParser(description="滚动窗口交叉验证")
    parser.add_argument("--days", type=int, default=90, help="数据总天数（示例数据）")
    parser.add_argument("--models", nargs="+", default=["physical", "xgboost"],
                        help="参与验证的模型（默认 physical+xgboost，跑得快）")
    parser.add_argument("--mode", type=str, default="expanding", choices=["expanding", "sliding"])
    parser.add_argument("--train-days", type=int, default=14, help="初始训练窗（sliding=固定窗宽）")
    parser.add_argument("--test-days", type=int, default=7, help="每折测试窗")
    parser.add_argument("--step-days", type=int, default=7, help="步长")
    parser.add_argument("--max-folds", type=int, default=None, help="最多折数")
    parser.add_argument("--save", type=str, default=None, help="报告保存路径")
    args = parser.parse_args()

    # 数据：示例数据（接InfluxDB后替换 _gen_sample_data 为真实查询）
    df = _gen_sample_data(n_days=args.days)

    # 清洗 + 特征
    cleaner = DataCleaner()
    df = cleaner.clean_weather_data(df)
    df = cleaner.clean_power_data(df)
    fe = FeatureEngineer()
    df = fe.build_features(df, include_lag=True, include_rolling=True)
    feature_cols = fe.get_feature_columns(df)

    # 模型工厂：每折一个全新实例，杜绝折间状态泄漏
    def make(model_type: str):
        def factory():
            if model_type == "physical":
                from models.physical_model import PhysicalModel
                return PhysicalModel(model_name="physical_baseline")
            if model_type == "xgboost":
                from models.xgboost_model import XGBoostModel
                return XGBoostModel(model_name="xgboost_v1")
            if model_type == "lstm":
                from models.lstm_model import LSTMModel
                return LSTMModel(model_name="lstm_v1", epochs=20)
            if model_type == "transformer":
                from models.transformer_model import TransformerModel
                return TransformerModel(model_name="transformer_v1", epochs=10)
            raise ValueError(model_type)
        return factory

    factories = {mt: make(mt) for mt in args.models}

    validator = RollingValidator(
        initial_train_days=args.train_days,
        test_window_days=args.test_days,
        step_days=args.step_days,
        mode=args.mode,
        max_folds=args.max_folds,
    )
    report = validator.validate(df, factories, feature_cols)

    # 打印汇总表
    print("\n" + "=" * 70)
    print("滚动窗口验证汇总（看 worst 折，那是你上线后最烂的一天）")
    print("=" * 70)
    for name, s in report["summary"].items():
        if s.get("status") == "all_failed":
            print(f"  {name:<20} 全部失败")
            continue
        print(f"  {name:<20} NMAE均值={s['nmae_mean']:.2%} ±{s['nmae_std']:.2%} "
              f"最差折={s['nmae_worst']:.2%} 等级={s['grade']} "
              f"漂移嫌疑={'⚠是' if s['drift_suspected'] else '否'}")
    print(f"\n按最差折排名的最优模型: {report['best_by_worst_fold']}")
    print("=" * 70)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"报告已保存: {args.save}")


if __name__ == "__main__":
    main()
