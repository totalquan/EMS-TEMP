# EMS-TEMP
The distributed photovoltaic power prediction and energy management system, based on NWP weather forecasting, covers the entire process from data access, feature engineering, model training, prediction services, MPC scheduling optimization, containerized deployment, to monitoring and operation and maintenance.

---

## pv-forecast-ems 运行结果

### 【1/4】系统配置加载
```
电站: 鄂尔多斯5MW分布式光伏电站
装机: 5000.0kW
位置: (39.61°N, 110.07°E), 海拔1300m
倾角: 38°, 系统效率: 0.83
预测: 16步 × 15min = 4小时
MPC: 4步时域, 储能1000kW/2000kWh
```

### 【2/4】物理模型预测
```
白天 (GHI≈650W/m², 25℃):
  预测功率范围: 2283 ~ 2693 kW
  典型值 (650W/m²): 2478 kW
夜间 (GHI=0):
  预测功率: 0 kW
```

### 【3/4】评估指标
```
NMAE: 0.75%   (目标<15%)  
合格率: 100%   (目标>80%)  
```

### 【4/4】MPC滚动优化（PuLP/CBC）
```
求解状态: Optimal
4步调度净收益: -287.55元 (负=净赚钱)

  步    光伏kW   负荷kW   购电kW   售电kW   充电kW   放电kW   SOC
  0    2564    2000      0     1514      0     1000    50.0%
  1    2549    2500      0      999      0     1000    61.9%
  2    2583    3000      0      533      0     1000    73.8%
  3    2690    2800      0      240      0      368    85.6%
SOC终值: 90.0%
```

**逻辑解读**：光伏大发（~2500kW）大于负荷（2000-3000kW），MPC自动选择：
1. 储能满功率充电（1000kW）消纳富余绿电
2. 剩余电力售给电网（1514→240kW，随负荷递增递减）
3. 不购电，SOC从50%充到90%上限

---

