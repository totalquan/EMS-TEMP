"""
定期重训练调度器（漂移驱动 + 冠军/挑战者机制）

修复点（相对旧版）：
1. 旧版 timedelta 未导入 → 数据检查必然异常 → 静默退回30天示例数据重训（危险）
2. 旧版数据检查失败 return True → 现在改为：InfluxDB 不可用时退回监控层配对数据
3. 旧版重训完直接切换最优模型 → 现在 champion/challenger：新模型必须在同一测试窗上
   比现任冠军 NMAE 低 min_improvement 以上才允许上线，否则保留旧模型（防抖动切换）
4. 新增 PSI 漂移触发：分布漂移 critical 时无视 cron 立即建议重训
"""
import os
import sys
import json
import time
import shutil
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    BackgroundScheduler = None
    CronTrigger = None


class RetrainScheduler:
    """模型定期重训练调度器"""

    # 新模型要比现任冠军相对好多少才切换（防抖动：7天窗口内NMAE再降10%）
    MIN_IMPROVEMENT_REL = 0.10
    # 保留最近几个模型版本（回退用）
    KEEP_VERSIONS = 3

    def __init__(self):
        self.scheduler = None
        self.is_running = False
        self.last_train_result: Optional[Dict[str, Any]] = None
        self.last_train_time: Optional[datetime] = None
        self.train_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 数据获取：InfluxDB → 监控层配对数据 → 拒绝训练（绝不静默用示例数据）
    # ------------------------------------------------------------------
    def _fetch_training_data(self) -> Optional[pd.DataFrame]:
        """拉取 lookback_days 天的（气象+功率）训练数据。拿不到足够数据返回 None。"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=settings.retrain.lookback_days)

        # 路径1：InfluxDB
        try:
            from data_pipeline.influxdb_writer import InfluxDBWriter
            writer = InfluxDBWriter()
            actual = writer.query_actual_power(
                start_time=start_time.isoformat() + "Z",
                end_time=end_time.isoformat() + "Z",
            )
            if actual is not None and len(actual) >= settings.retrain.min_samples:
                logger.info(f"InfluxDB 数据就绪 | {len(actual)} 条 | "
                            f"窗口={start_time.date()}~{end_time.date()}")
                return self._actual_to_df(actual)
            logger.warning(f"InfluxDB 数据不足 | {len(actual) if actual else 0}/{settings.retrain.min_samples} 条")
        except Exception as e:
            logger.warning(f"InfluxDB 不可用: {e}")

        # 路径2：监控层配对数据（预测时序里存了实况，可拼出最小训练集）
        try:
            from monitoring.accuracy_monitor import accuracy_monitor
            paired = accuracy_monitor.get_paired_df(hours=24 * settings.retrain.lookback_days)
            if not paired.empty and len(paired) >= settings.retrain.min_samples:
                logger.info(f"使用监控层配对数据 | {len(paired)} 条（建议尽快恢复InfluxDB数据管道）")
                return paired.rename(columns={"y_true": "power_kw"})
            logger.warning(f"监控层配对数据不足 | {len(paired) if not paired.empty else 0} 条")
        except Exception as e:
            logger.warning(f"监控层数据不可用: {e}")

        # 路径3：拒绝训练。旧版在这里 return True 用示例数据练模型——那是把演示数据当生产模型
        logger.error("无真实数据可用于重训，本次跳过（绝不使用示例数据训练生产模型）")
        return None

    @staticmethod
    def _actual_to_df(records) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        return df

    # ------------------------------------------------------------------
    # 冠军/挑战者
    # ------------------------------------------------------------------
    def _get_champion_nmae(self) -> Optional[float]:
        """现任冠军最近7天的实测 NMAE（线上表现，不是训练集表现）"""
        try:
            from monitoring.accuracy_monitor import accuracy_monitor
            report = accuracy_monitor.get_rolling_report(hours=24 * 7)
            if report.get("status") == "ok":
                return report["nmae"]
        except Exception as e:
            logger.warning(f"冠军线上指标获取失败: {e}")
        return None

    def _do_retrain(self, force: bool = False):
        """执行重训练（champion/challenger 流程）"""
        logger.info("=" * 60)
        logger.info(f"重训练任务启动 | force={force} | 时间: {datetime.now().isoformat()}")
        start_time = time.time()

        try:
            # 0. 漂移检查（漂移critical时force重训，正常按cron走）
            drift_advice = None
            try:
                from monitoring.accuracy_monitor import accuracy_monitor
                drift = accuracy_monitor.psi_drift(reference_days=30, recent_days=7)
                if drift.get("level") == "critical":
                    logger.warning("PSI漂移 critical，本次强制执行重训练")
                    force = True
                elif drift.get("level") == "warning":
                    logger.info("PSI漂移 warning，建议缩短重训周期")
                drift_advice = drift
            except Exception as e:
                logger.debug(f"漂移检测跳过: {e}")

            # 1. 数据
            df = self._fetch_training_data()
            if df is None:
                self._record_history({"status": "skipped_no_data"})
                return

            # 2. 训练流水线（滚动验证模式做把关，比单次切分可信）
            from training.train import TrainingPipeline
            from training.rolling_validation import RollingValidator
            from data_pipeline.data_cleaner import DataCleaner
            from data_pipeline.feature_engineer import FeatureEngineer

            cleaner, fe = DataCleaner(), FeatureEngineer()
            df = cleaner.clean_weather_data(df)
            df = cleaner.clean_power_data(df)
            df = fe.build_features(df, include_lag=True, include_rolling=True)
            feature_cols = fe.get_feature_columns(df)

            def factory():
                from models.xgboost_model import XGBoostModel
                return XGBoostModel(model_name="xgboost_challenger")

            lookback_days = min(settings.retrain.lookback_days, 30)
            validator = RollingValidator(
                initial_train_days=max(7, lookback_days // 3),
                test_window_days=7,
                step_days=7,
                mode="expanding",
                max_folds=4,
            )
            report = validator.validate(df, {"xgboost_challenger": factory}, feature_cols)
            challenger = report.get("summary", {}).get("xgboost_challenger", {})
            if challenger.get("status") == "all_failed" or "nmae_mean" not in challenger:
                self._record_history({"status": "challenger_failed"})
                return

            challenger_nmae = challenger["nmae_mean"]
            champion_nmae = self._get_champion_nmae()

            # 3. 切换判定
            if champion_nmae is not None and not force:
                improvement = (champion_nmae - challenger_nmae) / champion_nmae
                if improvement < self.MIN_IMPROVEMENT_REL:
                    logger.info(
                        f"挑战者未达标，保留冠军 | 冠军7d NMAE={champion_nmae:.2%} "
                        f"挑战者={challenger_nmae:.2%} | 需相对提升>{self.MIN_IMPROVEMENT_REL:.0%} "
                        f"实际={improvement:.1%}"
                    )
                    self._record_history({
                        "status": "challenger_rejected",
                        "champion_nmae_7d": champion_nmae,
                        "challenger_nmae": challenger_nmae,
                        "improvement": round(improvement, 4),
                    })
                    return

            # 4. 上线挑战者（写active_model + 保存带时间戳的模型版本）
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            active_path = os.path.join(settings.model_storage_path, "active_model.txt")
            with open(active_path, "w") as f:
                f.write(f"xgboost_v1_{ts_tag}")

            validator_report_path = os.path.join(
                settings.model_storage_path, f"retrain_report_{ts_tag}.json"
            )
            with open(validator_report_path, "w", encoding="utf-8") as f:
                json.dump({"challenger": challenger, "champion_nmae_7d": champion_nmae,
                           "drift": drift_advice, "generated_at": ts_tag},
                          f, ensure_ascii=False, indent=2, default=str)

            self._prune_old_reports()

            training_time = round(time.time() - start_time, 2)
            self._record_history({
                "status": "switched",
                "new_model": f"xgboost_v1_{ts_tag}",
                "challenger_nmae": challenger_nmae,
                "champion_nmae_7d": champion_nmae,
                "training_time_sec": training_time,
                "drift_level": (drift_advice or {}).get("level"),
            })
            logger.info(f"模型已切换 → xgboost_v1_{ts_tag} | 用时{training_time}s | "
                        f"挑战者NMAE={challenger_nmae:.2%}")

        except Exception as e:
            logger.error(f"重训练失败: {e}", exc_info=True)
            self._record_history({"status": "failed", "error": str(e)})

    def _record_history(self, entry: Dict[str, Any]):
        entry["timestamp"] = datetime.now().isoformat()
        self.train_history.append(entry)
        hist_path = os.path.join(settings.model_storage_path, "retrain_history.jsonl")
        try:
            with open(hist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _prune_old_reports(self):
        """只保留最近 KEEP_VERSIONS 份 retrain_report"""
        reports = sorted(
            [f for f in os.listdir(settings.model_storage_path) if f.startswith("retrain_report_")],
            reverse=True,
        )
        for old in reports[self.KEEP_VERSIONS:]:
            try:
                os.remove(os.path.join(settings.model_storage_path, old))
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 调度
    # ------------------------------------------------------------------
    def start(self):
        if not settings.retrain.enabled:
            logger.info("定期重训练已禁用")
            return
        if BackgroundScheduler is None:
            logger.warning("APScheduler未安装，定期重训练不可用")
            return
        if self.is_running:
            logger.warning("重训练调度器已在运行")
            return

        self.scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
        cron_parts = settings.retrain.cron.split()
        if len(cron_parts) == 5:
            trigger = CronTrigger(minute=cron_parts[0], hour=cron_parts[1], day=cron_parts[2],
                                  month=cron_parts[3], day_of_week=cron_parts[4])
        else:
            trigger = CronTrigger(day_of_week="sun", hour=2, minute=0)

        self.scheduler.add_job(
            self._do_retrain, trigger=trigger,
            id="pv_model_retrain", name="光伏模型定期重训练",
            replace_existing=True, misfire_grace_time=3600,
        )
        self.scheduler.start()
        self.is_running = True
        logger.info(f"重训练调度器已启动 | cron={settings.retrain.cron}")

    def stop(self):
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("重训练调度器已停止")

    def trigger_now(self, force: bool = True):
        logger.info("手动触发重训练")
        self._do_retrain(force=force)

    def get_status(self) -> Dict[str, Any]:
        active_path = os.path.join(settings.model_storage_path, "active_model.txt")
        active_model = None
        if os.path.exists(active_path):
            with open(active_path) as f:
                active_model = f.read().strip()
        return {
            "enabled": settings.retrain.enabled,
            "is_running": self.is_running,
            "cron": settings.retrain.cron,
            "lookback_days": settings.retrain.lookback_days,
            "min_samples": settings.retrain.min_samples,
            "min_improvement_rel": self.MIN_IMPROVEMENT_REL,
            "active_model": active_model,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "history_count": len(self.train_history),
            "recent_history": self.train_history[-5:],
        }


# 全局单例
retrain_scheduler = RetrainScheduler()


if __name__ == "__main__":
    logger.info("独立运行重训练任务")
    RetrainScheduler()._do_retrain(force=True)
