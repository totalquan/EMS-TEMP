"""
调度层 - MPC模型预测控制联合优化
基于光伏预测结果，优化储能充放电策略，最小化运行成本
支持Gurobi（商业求解器）和PuLP/CBC（开源求解器）双引擎
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from config.settings import settings


@dataclass
class MPCOptimizationConfig:
    """MPC优化配置"""
    # 预测 horizon
    horizon_steps: int = 16  # 4小时 × 15分钟 = 16步
    dt_hours: float = 0.25   # 时间步长（小时）

    # 光伏
    pv_capacity_kw: float = 5000.0

    # 储能
    ess_capacity_kwh: float = 2000.0
    ess_max_power_kw: float = 500.0
    ess_soc_min: float = 0.2
    ess_soc_max: float = 0.8
    ess_soc_init: float = 0.5
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    ess_degradation_cost: float = 0.05  # 元/kWh

    # 负荷
    load_kw: np.ndarray = field(default_factory=lambda: np.ones(16) * 3000.0)

    # 电价
    price_buy: np.ndarray = field(default_factory=lambda: np.ones(16) * 0.8)  # 购电价
    price_sell: float = 0.3  # 售电价

    # 电网
    grid_buy_max_kw: float = 2000.0
    grid_sell_max_kw: float = 1000.0


class MPCOptimizer:
    """
    MPC联合优化器
    目标：最小化（购电成本 - 售电收入 + 储能损耗成本）
    约束：功率平衡、储能SOC、电网功率上下限
    """

    def __init__(self, config: Optional[MPCOptimizationConfig] = None):
        self.config = config or MPCOptimizationConfig()
        self.solver_engine = settings.solver.engine
        logger.info(f"MPC优化器初始化 | 求解器={self.solver_engine} | horizon={self.config.horizon_steps}步")

    def optimize(self, pv_forecast_kw: np.ndarray,
                 load_kw: Optional[np.ndarray] = None,
                 price_buy: Optional[np.ndarray] = None,
                 soc_init: Optional[float] = None) -> Dict[str, Any]:
        """
        执行MPC优化
        :param pv_forecast_kw: 光伏预测功率（horizon_steps维）
        :param load_kw: 负荷功率
        :param price_buy: 购电电价
        :param soc_init: 初始SOC
        :return: 优化结果
        """
        cfg = self.config
        N = cfg.horizon_steps

        # 输入校验与默认值
        pv = np.array(pv_forecast_kw[:N], dtype=float)
        if len(pv) < N:
            pv = np.pad(pv, (0, N - len(pv)), mode="edge")

        load = np.array(load_kw[:N], dtype=float) if load_kw is not None else cfg.load_kw[:N]
        if len(load) < N:
            load = np.pad(load, (0, N - len(load)), mode="edge")

        price = np.array(price_buy[:N], dtype=float) if price_buy is not None else cfg.price_buy[:N]
        if len(price) < N:
            price = np.pad(price, (0, N - len(price)), mode="edge")

        soc0 = soc_init if soc_init is not None else cfg.ess_soc_init

        logger.info(f"开始MPC优化 | N={N} | 光伏均值={pv.mean():.1f}kW | 负荷均值={load.mean():.1f}kW")

        # 根据求解器引擎选择实现
        if self.solver_engine == "gurobi":
            result = self._optimize_gurobi(pv, load, price, soc0)
        else:
            result = self._optimize_pulp(pv, load, price, soc0)

        return result

    def _optimize_pulp(self, pv: np.ndarray, load: np.ndarray,
                        price: np.ndarray, soc0: float) -> Dict[str, Any]:
        """PuLP/CBC开源求解器实现"""
        try:
            import pulp
        except ImportError:
            logger.error("PuLP未安装，回退到启发式策略")
            return self._heuristic_strategy(pv, load, price, soc0)

        cfg = self.config
        N = cfg.horizon_steps
        dt = cfg.dt_hours

        # 创建问题
        prob = pulp.LpProblem("PV_ESS_MPC", pulp.LpMinimize)

        # 决策变量
        P_ch = [pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=cfg.ess_max_power_kw) for t in range(N)]
        P_dis = [pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=cfg.ess_max_power_kw) for t in range(N)]
        P_grid_buy = [pulp.LpVariable(f"P_grid_buy_{t}", lowBound=0, upBound=cfg.grid_buy_max_kw) for t in range(N)]
        P_grid_sell = [pulp.LpVariable(f"P_grid_sell_{t}", lowBound=0, upBound=cfg.grid_sell_max_kw) for t in range(N)]
        SOC = [pulp.LpVariable(f"SOC_{t}", lowBound=cfg.ess_soc_min * cfg.ess_capacity_kwh,
                                upBound=cfg.ess_soc_max * cfg.ess_capacity_kwh) for t in range(N + 1)]

        # 目标函数：最小化总成本
        cost = pulp.lpSum([
            price[t] * P_grid_buy[t] * dt - cfg.price_sell * P_grid_sell[t] * dt
            + cfg.ess_degradation_cost * (P_ch[t] + P_dis[t]) * dt
            for t in range(N)
        ])
        prob += cost

        # 约束1：初始SOC
        prob += SOC[0] == soc0 * cfg.ess_capacity_kwh

        # 约束2：SOC动态
        for t in range(N):
            prob += SOC[t + 1] == SOC[t] + (P_ch[t] * cfg.eta_ch - P_dis[t] / cfg.eta_dis) * dt

        # 约束3：功率平衡
        for t in range(N):
            prob += pv[t] + P_grid_buy[t] + P_dis[t] == load[t] + P_ch[t] + P_grid_sell[t]

        # 约束4：充放电互斥（大M法）
        M = cfg.ess_max_power_kw + 1
        for t in range(N):
            prob += P_ch[t] <= M  # 已由上下界约束
            prob += P_dis[t] <= M

        # 求解
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=settings.solver.time_limit)
        status = prob.solve(solver)

        if pulp.LpStatus[status] != "Optimal":
            logger.warning(f"PuLP求解状态: {pulp.LpStatus[status]}，使用启发式策略")
            return self._heuristic_strategy(pv, load, price, soc0)

        # 提取结果
        result = self._extract_results(
            pv, load, price, N, dt,
            [pulp.value(P_ch[t]) for t in range(N)],
            [pulp.value(P_dis[t]) for t in range(N)],
            [pulp.value(P_grid_buy[t]) for t in range(N)],
            [pulp.value(P_grid_sell[t]) for t in range(N)],
            [pulp.value(SOC[t]) for t in range(N + 1)],
            pulp.value(prob.objective),
        )
        result["solver"] = "PuLP/CBC"
        return result

    def _optimize_gurobi(self, pv: np.ndarray, load: np.ndarray,
                          price: np.ndarray, soc0: float) -> Dict[str, Any]:
        """Gurobi商业求解器实现"""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            logger.warning("Gurobi未安装，回退到PuLP")
            return self._optimize_pulp(pv, load, price, soc0)

        cfg = self.config
        N = cfg.horizon_steps
        dt = cfg.dt_hours

        try:
            m = gp.Model("PV_ESS_MPC")
            m.setParam("OutputFlag", 0)
            m.setParam("TimeLimit", settings.solver.time_limit)

            # 决策变量
            P_ch = m.addVars(N, lb=0, ub=cfg.ess_max_power_kw, name="P_ch")
            P_dis = m.addVars(N, lb=0, ub=cfg.ess_max_power_kw, name="P_dis")
            P_grid_buy = m.addVars(N, lb=0, ub=cfg.grid_buy_max_kw, name="P_grid_buy")
            P_grid_sell = m.addVars(N, lb=0, ub=cfg.grid_sell_max_kw, name="P_grid_sell")
            SOC = m.addVars(N + 1, lb=cfg.ess_soc_min * cfg.ess_capacity_kwh,
                            ub=cfg.ess_soc_max * cfg.ess_capacity_kwh, name="SOC")

            # 目标函数
            m.setObjective(
                gp.quicksum(price[t] * P_grid_buy[t] * dt - cfg.price_sell * P_grid_sell[t] * dt
                           + cfg.ess_degradation_cost * (P_ch[t] + P_dis[t]) * dt
                           for t in range(N)),
                GRB.MINIMIZE
            )

            # 约束
            m.addConstr(SOC[0] == soc0 * cfg.ess_capacity_kwh, "init_soc")
            for t in range(N):
                m.addConstr(SOC[t + 1] == SOC[t] + (P_ch[t] * cfg.eta_ch - P_dis[t] / cfg.eta_dis) * dt, f"soc_dyn_{t}")
                m.addConstr(pv[t] + P_grid_buy[t] + P_dis[t] == load[t] + P_ch[t] + P_grid_sell[t], f"power_balance_{t}")

            m.optimize()

            if m.status != GRB.OPTIMAL:
                logger.warning(f"Gurobi求解状态: {m.status}，使用启发式策略")
                return self._heuristic_strategy(pv, load, price, soc0)

            result = self._extract_results(
                pv, load, price, N, dt,
                [P_ch[t].X for t in range(N)],
                [P_dis[t].X for t in range(N)],
                [P_grid_buy[t].X for t in range(N)],
                [P_grid_sell[t].X for t in range(N)],
                [SOC[t].X for t in range(N + 1)],
                m.objVal,
            )
            result["solver"] = "Gurobi"
            return result

        except Exception as e:
            logger.error(f"Gurobi求解异常: {e}，回退到启发式策略")
            return self._heuristic_strategy(pv, load, price, soc0)

    def _extract_results(self, pv, load, price, N, dt,
                         P_ch, P_dis, P_grid_buy, P_grid_sell, SOC, total_cost) -> Dict[str, Any]:
        """提取并格式化优化结果"""
        schedule = []
        for t in range(N):
            net_load = load[t] - pv[t]  # 净负荷（正=需要供电，负=富余）
            schedule.append({
                "step": t,
                "time_offset_min": t * int(dt * 60),
                "pv_kw": round(float(pv[t]), 2),
                "load_kw": round(float(load[t]), 2),
                "net_load_kw": round(float(net_load), 2),
                "ess_charge_kw": round(float(P_ch[t]), 2),
                "ess_discharge_kw": round(float(P_dis[t]), 2),
                "ess_net_kw": round(float(P_ch[t] - P_dis[t]), 2),
                "grid_buy_kw": round(float(P_grid_buy[t]), 2),
                "grid_sell_kw": round(float(P_grid_sell[t]), 2),
                "grid_net_kw": round(float(P_grid_buy[t] - P_grid_sell[t]), 2),
                "soc_kwh": round(float(SOC[t]), 2),
                "soc_percent": round(float(SOC[t]) / self.config.ess_capacity_kwh * 100, 2),
                "price_buy": round(float(price[t]), 4),
                "step_cost": round(float(price[t] * P_grid_buy[t] * dt - self.config.price_sell * P_grid_sell[t] * dt), 4),
            })

        result = {
            "status": "optimal",
            "solver": "unknown",
            "horizon_steps": N,
            "time_step_min": int(dt * 60),
            "total_cost_yuan": round(float(total_cost), 4),
            "total_grid_buy_kwh": round(float(sum(P_grid_buy) * dt), 2),
            "total_grid_sell_kwh": round(float(sum(P_grid_sell) * dt), 2),
            "total_ess_charge_kwh": round(float(sum(P_ch) * dt), 2),
            "total_ess_discharge_kwh": round(float(sum(P_dis) * dt), 2),
            "final_soc_percent": round(float(SOC[N]) / self.config.ess_capacity_kwh * 100, 2),
            "schedule": schedule,
        }
        logger.info(f"MPC优化完成 | 总成本={result['total_cost_yuan']:.2f}元 | "
                    f"购电={result['total_grid_buy_kwh']:.1f}kWh | 售电={result['total_grid_sell_kwh']:.1f}kWh")
        return result

    def _heuristic_strategy(self, pv: np.ndarray, load: np.ndarray,
                             price: np.ndarray, soc0: float) -> Dict[str, Any]:
        """
        启发式兜底策略（求解器不可用时使用）
        策略：谷时充电、峰时放电、富余光伏充电
        """
        cfg = self.config
        N = cfg.horizon_steps
        dt = cfg.dt_hours

        P_ch = np.zeros(N)
        P_dis = np.zeros(N)
        P_grid_buy = np.zeros(N)
        P_grid_sell = np.zeros(N)
        SOC = np.zeros(N + 1)
        SOC[0] = soc0 * cfg.ess_capacity_kwh

        for t in range(N):
            net = load[t] - pv[t]  # 净负荷

            if net > 0:  # 需要供电
                # 优先储能放电
                dis_avail = min(cfg.ess_max_power_kw,
                                (SOC[t] - cfg.ess_soc_min * cfg.ess_capacity_kwh) * cfg.eta_dis / dt)
                P_dis[t] = min(net, dis_avail)
                remaining = net - P_dis[t]
                # 剩余从电网买
                P_grid_buy[t] = min(remaining, cfg.grid_buy_max_kw)
            else:  # 光伏富余
                surplus = -net
                # 优先充电
                ch_avail = min(cfg.ess_max_power_kw,
                                (cfg.ess_soc_max * cfg.ess_capacity_kwh - SOC[t]) / (cfg.eta_ch * dt))
                P_ch[t] = min(surplus, ch_avail)
                remaining = surplus - P_ch[t]
                # 剩余卖电
                P_grid_sell[t] = min(remaining, cfg.grid_sell_max_kw)

            # 更新SOC
            SOC[t + 1] = SOC[t] + (P_ch[t] * cfg.eta_ch - P_dis[t] / cfg.eta_dis) * dt
            SOC[t + 1] = np.clip(SOC[t + 1], cfg.ess_soc_min * cfg.ess_capacity_kwh,
                                  cfg.ess_soc_max * cfg.ess_capacity_kwh)

        total_cost = float(sum(price[t] * P_grid_buy[t] * dt - cfg.price_sell * P_grid_sell[t] * dt
                                + cfg.ess_degradation_cost * (P_ch[t] + P_dis[t]) * dt
                                for t in range(N)))

        result = self._extract_results(pv, load, price, N, dt,
                                        P_ch, P_dis, P_grid_buy, P_grid_sell, SOC, total_cost)
        result["solver"] = "heuristic"
        result["status"] = "heuristic_fallback"
        return result
