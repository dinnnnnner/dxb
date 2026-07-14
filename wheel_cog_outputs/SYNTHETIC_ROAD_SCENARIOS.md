# 非爆胎道路与故障场景模拟数据

`generate_synthetic_road_scenarios.py` 从真实爆胎发生前的正常四轮轮速中截取基线，注入 8 类可复现扰动，用来检查爆胎报警算法对 hard negative 和渐变故障的鲁棒性。

## 生成数据

在仓库根目录运行：

```bash
python wheel_cog_outputs/generate_synthetic_road_scenarios.py \
  --samples-per-scenario 10
```

默认输出到 `wheel_cog_outputs/synthetic_road_scenario_dataset/`：

- `samples/*.csv`：50 秒、约 100 Hz 的四轮轮速样本，可由现有报警脚本按 `corrected` 序列读取。
- `manifest.csv`：场景、中文名称、来源、发生时间、目标轮、变体、强度和完整参数。
- `dataset_config.json`：生成参数、随机种子、轮位映射和各场景数量。

已有输出时，脚本会拒绝覆盖。确认需要重新生成时显式增加 `--overwrite`。

只生成部分场景：

```bash
python wheel_cog_outputs/generate_synthetic_road_scenarios.py \
  --scenarios hard_brake sharp_turn pothole speed_bump \
  --samples-per-scenario 20 \
  --output-dir /tmp/road_events
```

可通过 `--seed` 固定随机结果。`--scenario-time-s` 默认为 40 秒，以便当前算法先完成约 30 秒预热；低胎压是从样本开始就存在的稳态工况，`scenario_start_s` 固定为 0。

急转弯默认使用 `2.75 m` 轴距、`1.60 m` 轮距和 `0.33 m` 轮胎有效半径，并在 `2.0～5.0 m/s²` 范围内随机生成峰值横向加速度。可按实车修改：

```bash
python wheel_cog_outputs/generate_synthetic_road_scenarios.py \
  --wheelbase-m 2.85 \
  --track-width-m 1.65 \
  --tire-radius-m 0.34 \
  --turn-lateral-accel-min-m-s2 2.0 \
  --turn-lateral-accel-max-m-s2 4.5 \
  --overwrite
```

打滑采用纵向滑移率模型：

```text
slip_ratio = (车轮圆周速度 - 车辆参考速度)
             / max(|车轮圆周速度|, |车辆参考速度|)
```

车辆参考速度由另外三个未打滑车轮的中位数估计。驱动空转依次经历滑移建立、峰值、TCS 控制平台和恢复；制动滑移依次经历抱死趋势、峰值、ABS 控制平台和恢复。样本编号奇偶交替生成这两种情况。默认按后驱车选择空转目标轮，可用 `--driven-axle front|rear|all` 调整。

## 场景模型

| 场景 | 注入的轮速特征 |
|---|---|
| 急刹 `hard_brake` | 四轮共同减速，叠加前后轴强度不同的 ABS 周期滑移 |
| 急转弯 `sharp_turn` | 由车速和横向加速度计算瞬时转弯半径，再按 Ackermann 几何计算四轮路径半径和轮速 |
| 坑洼 `pothole` | 单轮短时负向冲击和衰减振荡，同轴另一轮有弱耦合 |
| 减速带 `speed_bump` | 前轴先冲击、后轴延迟冲击，两侧轮有轻微不对称 |
| 低胎压 `low_tire_pressure` | 单轮有效滚动半径持续减小，对应角速度持续偏高并缓慢漂移 |
| 慢漏气 `slow_leak` | 单轮有效半径随时间平滑减小；在 50 秒窗口中进行了时间压缩 |
| 打滑 `slip` | 以另外三轮估计车速，按纵向滑移率生成单轮空转或制动滑移，并模拟 TCS/ABS 建立、控制和恢复阶段 |
| 传感器异常 `sensor_anomaly` | 循环生成尖峰、掉零、卡死、阶跃偏置和噪声突发等变体 |

轮位沿用当前工程假设：`wheel0=FL`、`wheel1=FR`、`wheel2=RL`、`wheel3=RR`。

## 用当前算法评价

这些场景都不是爆胎，因此 `manifest.csv` 中保留兼容字段 `sample_type=normal`，同时明确写入 `expected_blowout=0`。安装项目依赖后可直接运行：

```bash
python wheel_cog_outputs/evaluate_augmented_fast_batch_display.py \
  --dataset-dir wheel_cog_outputs/synthetic_road_scenario_dataset \
  --html-mode hard
```

评价时按 `source_event_id` 分组隔离，避免同一真实行驶片段同时进入调参与验证集合。

## 使用边界

这些数据只模拟四轮轮速通道的典型信号形态，不是整车动力学、悬架、轮胎压力、IMU 和轮速传感器的联合高保真仿真。参数范围用于算法压力测试，不能替代实车采集，也不能据此声明真实道路误报率。特别是慢漏气在短样本内做了时间压缩，低胎压用有效滚动半径变化近似；正式验证仍需采集对应的真实场景数据。
