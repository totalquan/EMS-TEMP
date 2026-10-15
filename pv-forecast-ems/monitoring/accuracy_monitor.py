"""
监控层 - 预测精度监控器（MAPE/NMAE 拖尾 + 分级告警 + 分布漂移检测）

设计原则：
1. 不依赖 InfluxDB 也能跑——默认落 JSONL 文件（部署了 InfluxDB 可平滑替换存储层）
2. 预测值和实际值分开记录，事后配对计算（实际值总是晚到，这是常态）
3. 三级告警：ok / warning / critical，阈值对齐电网"两个细则"考核口径
4. PSI 漂移检测：对比"参考期"与"最近期"的出力分布，检测季节/积灰/装机变化

用法：
    from monitoring.accuracy_monitor import accuracy_monitor
    accuracy_monitor.record_predictions(timestamps, y_pred, model="lstm_v1")
    accuracy_monitor.record_actuals(timestamps, y_true)
    report = accuracy_monitor.get_rolling_report(hours=24)  # 含告警状态
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


class AccuracyMonitor:
    """预测精度监控器"""

    # 告警阈值（NMAE口径，对齐两个细则：日前 NMAE≤15% 合格）
    NMAE_WARNING = 0.15
    NMAE_CRITICAL = 0.20
    # 合格率阈值
    QR_WARNING = 0.80
    QR_CRITICAL = 0.70
    # PSI 漂移阈值：>0.25 显著漂移，>0.1 轻度漂移
    PSI_WARNING = 0.10
    PSI_CRITICAL = 0.25

    def __init__(self, store_dir: Optional[str] = None):
        self.store_dir = store_dir or os.path.join(str(settings.model_storage_path), "monitoring")
        os.makedirs(self.store_dir, exist_ok=True)
        self.pred_path = os.path.join(self.store_dir, "predictions.jsonl")
        self.actual_path = os.path.join(self.store_dir, "actuals.jsonl")
        self.alert_path = os.path.join(self.store_dir, "alerts.jsonl")
        self.report_path = os.path.join(self.store_dir, "daily_reports.jsonl")

    # ------------------------------------------------------------------
    # 记录（预测侧 / 实际侧分离）
    # ------------------------------------------------------------------
    def record_predictions(self, timestamps: List, y_pred: List[float], model: str = "unknown"):
        """预测产生时调用。API 的 /forecast 路由出结果后立即调这里。"""
        records = [{"ts": pd.Timestamp(t).isoformat(), "y_pred": float(p), "model": model}
                   for t, p in zip(timestamps, y_pred)]
        self._append_jsonl(self.pred_path, records)
        logger.debug(f"记录预测 {len(records)} 条 | 模型={model}")

    def record_actuals(self, timestamps: List, y_true: List[float]):
        """实际功率到达时调用（SCADA/电表数据落库后）。同 ts 重复记录会覆盖。"""
        records = [{"ts": pd.Timestamp(t).isoformat(), "y_true": float(v)} for t, v in zip(timestamps, y_true)]
        self._append_jsonl(self.actual_path, records)

    # ------------------------------------------------------------------
    # 核心报表
    # ------------------------------------------------------------------
    def get_paired_df(self, hours: Optional[int] = None) -> pd.DataFrame:
        """按时间戳配对预测与实际（实际值为准做左连接，未配对的预测先跳过）"""
        pred = self._read_jsonl(self.pred_path)
        actual = self._read_jsonl(self.actual_path)
        if pred.empty or actual.empty:
            return pd.DataFrame()
        pred["ts"] = pd.to_datetime(pred["ts"])
        actual["ts"] = pd.to_datetime(actual["ts"])
        df = pd.merge(actual, pred, on="ts", how="inner")
        if hours:
            cutoff = df["ts"].max() - pd.Timedelta(hours=hours)
            df = df[df["ts"] >= cutoff]
        return df.sort_values("ts")

    def get_rolling_report(self, hours: int = 24, daytime_only: bool = True) -> Dict[str, Any]:
        """
        滚动窗口精度报告（默认24h）——这就是"每天看的 MAPE 拖尾曲线"的单点值
        :param hours: 滚动窗口小时数
        """
        df = self.get_paired_df(hours=hours)
        if df.empty:
            return {"status": "no_data", "window_hours": hours,
                    "message": "暂无配对数据，需预测与实际都积累后计算"}

        mask = df["ts"].dt.hour.between(6, 18) if daytime_only else slice(None)
        d = df[mask] if daytime_only else df
        if d.empty:
            d = df

        cap = settings.pv.capacity_kw
        err = (d["y_true"] - d["y_pred"]).abs()
        nmae = float(err.mean() / cap)
        qualified = float((err < 0.2 * cap).mean())
        # MAPE 只统计 >1% 装机的样本（夜间/低出力 MAPE 无意义）
        m = d["y_true"] > cap * 0.01
        mape = float((err[m] / d["y_true"][m]).mean()) if m.any() else None
        bias = float((d["y_pred"] - d["y_true"]).mean())

        report = {
            "status": "ok",
            "window_hours": hours,
            "sample_count": len(d),
            "generated_at": datetime.now().isoformat(),
            "nmae": nmae,
            "mape": mape,
            "qualified_rate": qualified,
            "bias_kw": bias,
            "models_included": df["model"].unique().tolist(),
        }
        # 告警判定
        report["alert"] = self._judge_alert(nmae, qualified)
        if report["alert"]["level"] != "ok":
            self._emit_alert(report)
        return report

    def get_daily_series(self, days: int = 14) -> pd.DataFrame:
        """按天聚合的精度序列——拖尾曲线的数据源，画图/看趋势用"""
        df = self.get_paired_df(hours=24 * days)
        if df.empty:
            return pd.DataFrame()
        cap = settings.pv.capacity_kw
        df["date"] = df["ts"].dt.date
        df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
        df["bias_raw"] = df["y_pred"] - df["y_true"]

        daily = df.groupby("date").agg(
            nmae=("abs_err", lambda e: e.mean() / cap),
            mae=("abs_err", "mean"),
            sample_count=("abs_err", "count"),
            bias_kw=("bias_raw", "mean"),
        )
        daily["alert_level"] = daily["nmae"].apply(lambda n: self._judge_alert(n, 1.0)["level"])
        return daily

    # ------------------------------------------------------------------
    # 漂移检测（PSI：Population Stability Index）
    # ------------------------------------------------------------------
    def psi_drift(self, reference_days: int = 30, recent_days: int = 7, bins: int = 10) -> Dict[str, Any]:
        """
        出力分布漂移检测：参考期 vs 最近期。
        场景：季节更替、组件积灰/衰减、新增装机、限电模式变化。
        PSI > 0.25 → 建议立即重训；0.10~0.25 → 缩短重训周期。
        """
        ref = self.get_paired_df(hours=24 * reference_days)["y_true"]
        rec = self.get_paired_df(hours=24 * recent_days)["y_true"]
        if len(ref) < 100 or len(rec) < 50:
            return {"status": "insufficient_data",
                    "message": f"参考期{len(ref)}条/最近期{len(rec)}条，样本不足"}

        cap = settings.pv.capacity_kw
        edges = np.linspace(0, cap, bins + 1)
        ref_hist, _ = np.histogram(ref, bins=edges)
        rec_hist, _ = np.histogram(rec, bins=edges)

        psi = self._psi_from_hist(ref_hist, rec_hist)
        if psi > self.PSI_CRITICAL:
            level, advice = "critical", "分布显著漂移，建议立即触发重训练"
        elif psi > self.PSI_WARNING:
            level, advice = "warning", "轻度漂移，建议缩短重训周期（如周→日）"
        else:
            level, advice = "ok", "分布稳定"

        result = {"status": "ok", "psi": round(float(psi), 4), "level": level,
                  "advice": advice, "reference_days": reference_days, "recent_days": recent_days}
        if level != "ok":
            self._emit_alert({"type": "drift", **result})
        return result

    @staticmethod
    def _psi_from_hist(ref_hist: np.ndarray, rec_hist: np.ndarray) -> float:
        """PSI = Σ (实际占比-参考占比) * ln(实际占比/参考占比)"""
        ref_ratio = ref_hist / max(ref_hist.sum(), 1)
        rec_ratio = rec_hist / max(rec_hist.sum(), 1)
        ref_ratio = np.clip(ref_ratio, 1e-6, None)
        rec_ratio = np.clip(rec_ratio, 1e-6, None)
        return float(np.sum((rec_ratio - ref_ratio) * np.log(rec_ratio / ref_ratio)))

    # ------------------------------------------------------------------
    # 告警
    # ------------------------------------------------------------------
    def _judge_alert(self, nmae: float, qualified_rate: float) -> Dict[str, Any]:
        if nmae > self.NMAE_CRITICAL or qualified_rate < self.QR_CRITICAL:
            return {"level": "critical", "reason": f"NMAE={nmae:.2%} 超过临界阈值{self.NMAE_CRITICAL:.0%}",
                    "action": "立即排查：数据质量→特征→模型，必要时回退上一个模型版本"}
        if nmae > self.NMAE_WARNING or qualified_rate < self.QR_WARNING:
            return {"level": "warning", "reason": f"NMAE={nmae:.2%} 超过警告阈值{self.NMAE_WARNING:.0%}",
                    "action": "关注3天趋势，持续超标则触发提前重训"}
        return {"level": "ok", "reason": "正常", "action": ""}

    def _emit_alert(self, report: Dict[str, Any]):
        alert = {"ts": datetime.now().isoformat(), **report}
        self._append_jsonl(self.alert_path, [alert])
        logger.warning(f"精度告警 | level={report.get('alert', report).get('level', report.get('level'))} "
                       f"| {json.dumps(report, ensure_ascii=False, default=str)[:300]}")

    def get_alerts(self, last_n: int = 20) -> List[Dict]:
        df = self._read_jsonl(self.alert_path)
        if df.empty:
            return []
        return json.loads(df.tail(last_n).to_json(orient="records", force_ascii=False))

    # ------------------------------------------------------------------
    # Prometheus 导出（对接 /metrics）
    # ------------------------------------------------------------------
    def prometheus_text(self) -> str:
        r24 = self.get_rolling_report(hours=24)
        drift = self.psi_drift()
        lines = []
        if r24.get("status") == "ok":
            lines += [
                "# HELP pv_forecast_nmae_24h Rolling 24h NMAE",
                "# TYPE pv_forecast_nmae_24h gauge",
                f"pv_forecast_nmae_24h {r24['nmae']:.6f}",
                "# HELP pv_forecast_qualified_rate_24h Rolling 24h qualified rate",
                "# TYPE pv_forecast_qualified_rate_24h gauge",
                f"pv_forecast_qualified_rate_24h {r24['qualified_rate']:.6f}",
            ]
        if drift.get("status") == "ok":
            lines += [
                "# HELP pv_forecast_psi_drift PSI drift between reference and recent window",
                "# TYPE pv_forecast_psi_drift gauge",
                f"pv_forecast_psi_drift {drift['psi']:.6f}",
            ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 存储工具
    # ------------------------------------------------------------------
    @staticmethod
    def _append_jsonl(path: str, records: List[Dict]):
        with open(path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _read_jsonl(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(rows)


# 全局单例
accuracy_monitor = AccuracyMonitor()


# ----------------------------------------------------------------------
# CLI：python monitoring/accuracy_monitor.py --demo
# ----------------------------------------------------------------------
def _demo():
    """用模拟数据演示完整监控闭环：记录预测 → 记录实际 → 报表 → 漂移 → 告警"""
    import shutil
    # 每次用带时间戳的新目录演示（不删除历史目录）
    store = os.path.join(
        str(settings.model_storage_path),
        f"monitoring_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(store, exist_ok=True)
    mon = AccuracyMonitor(store_dir=store)

    cap = settings.pv.capacity_kw
    # 模拟 20 天预测+实际：前 15 天误差小（正常），后 5 天实际分布下移+误差增大（模拟漂移/劣化）
    dates = pd.date_range(end=datetime.now(), periods=20 * 96, freq="15min")
    rng = np.random.default_rng(42)
    hour = dates.hour + dates.minute / 60
    base = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1) * cap * 0.6
    drift_factor = np.where(dates >= dates[-5 * 96], 0.75, 1.0)          # 后5天出力整体下降25%
    y_true = base * drift_factor * (0.95 + 0.1 * rng.standard_normal(len(dates)))
    y_pred = base * (0.95 + 0.1 * rng.standard_normal(len(dates)))       # 模型没跟上漂移

    mon.record_predictions(dates.tolist(), y_pred.tolist(), model="lstm_v1_demo")
    mon.record_actuals(dates.tolist(), y_true.tolist())

    print("\n【24h 滚动报告】")
    print(json.dumps(mon.get_rolling_report(hours=24), ensure_ascii=False, indent=2, default=str))
    print("\n【7d 滚动报告】")
    print(json.dumps(mon.get_rolling_report(hours=24 * 7), ensure_ascii=False, indent=2, default=str))
    print("\n【漂移检测（参考期30d vs 最近7d）】")
    print(json.dumps(mon.psi_drift(reference_days=30, recent_days=7), ensure_ascii=False, indent=2, default=str))
    print("\n【每日精度序列（拖尾曲线数据）】")
    daily = mon.get_daily_series(days=20)
    print(daily.to_string())
    print("\n【告警记录】")
    print(json.dumps(mon.get_alerts(), ensure_ascii=False, indent=2, default=str))
    print("\n【Prometheus 输出】")
    print(mon.prometheus_text())
    print(f"\n数据目录: {store}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="跑模拟演示")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        print(json.dumps(accuracy_monitor.get_rolling_report(hours=24),
                         ensure_ascii=False, indent=2, default=str))
