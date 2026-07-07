# alarm_detection 流程说明

本文档对应实现文件：

- `wheel_cog_outputs/alarm_detection_local_fast.py`

算法名称：

- `fast_ewma_leaky_evidence`

整体目标是从四轮轮速序列中提取异常残差，并用带泄漏的证据累计器输出报警状态。流程可以分成两类模块：

- 线性滤波/特征提取模块：可以用差分方程和 z 变换描述。
- 非线性报警判决模块：包含限幅、绝对值、阈值、滞回和状态机，不能用单个 z 域传递函数完整表示。

## 1. 输入与输出

### 输入

输入 CSV 至少需要以下列：

```text
time_s
wheel0_<series>_rad_s
wheel1_<series>_rad_s
wheel2_<series>_rad_s
wheel3_<series>_rad_s
```

其中 `series` 可选：

```text
raw
corrected
ref_comp_on
```

默认使用 `corrected`。

每一帧输入记为：

```text
w[n] = [w0[n], w1[n], w2[n], w3[n]]
```

平均轮速：

```text
avg[n] = (w0[n] + w1[n] + w2[n] + w3[n]) / 4
```

当：

```text
avg[n] >= min_avg_speed
```

该帧才进入正常报警检测逻辑。默认 `min_avg_speed = 20.0`。

### 输出

每帧输出一个 `DetectionResult`，写入 CSV 时包含：

```text
time_s
wheel0_rad_s
wheel1_rad_s
wheel2_rad_s
wheel3_rad_s
avg_speed_rad_s
combind
legacy_combind
wheel_feature
feature_baseline
innovation
alarm
score
off_threshold
on_threshold
evidence
signed_evidence
enter_threshold
exit_threshold
noise
recovery_active
recovery_frames_left
alarm_wheel
alarm_wheel_dev
```

其中：

- `alarm` 是最终报警状态，0 或 1。
- `combind` 是最终用于检测的特征值。
- `legacy_combind` 是基于四轮差分组合得到的主特征。
- `wheel_feature` 是单轮相对偏差特征。
- `innovation` 是 `combind` 相对自适应中心线的偏差。
- `evidence` / `score` 是报警证据累计值。
- `signed_evidence` 保留证据方向。
- `noise` 是自适应噪声估计。
- `enter_threshold` / `exit_threshold` 是动态进入/退出阈值。

## 2. 总体流程

单帧处理流程如下：

```text
四轮轮速 w[n]
  -> 平均速度门限判断
  -> 快速 EWMA fast[n]
  -> 慢速 EWMA slow[n]
  -> 四轮差分归一化
  -> fast/slow 残差组合 raw[n]
  -> EWMA 平滑 filtered[n]
  -> 输出缩放 legacy_combind[n]
  -> 可选单轮偏差 wheel_feature[n]
  -> 选择检测特征 combind[n]
  -> 自适应中心线 center[n]
  -> innovation[n] = combind[n] - center[n]
  -> 自适应噪声 noise[n]
  -> 动态阈值 enter/exit
  -> 正/负方向泄漏证据累计
  -> alarm 状态机
  -> recovery 恢复逻辑
```

## 3. EWMA 基本形式与 z 变换

代码中多处使用 EWMA：

```text
y[n] = y[n-1] + alpha * (x[n] - y[n-1])
```

等价于：

```text
y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
```

在零初始条件下，z 变换为：

```text
Y(z) = alpha * X(z) + (1 - alpha) * z^-1 * Y(z)
```

所以传递函数：

```text
H(z) = Y(z) / X(z)
     = alpha / (1 - (1 - alpha) z^-1)
```

这是一个一阶 IIR 低通滤波器。

极点：

```text
z = 1 - alpha
```

只要：

```text
0 < alpha < 2
```

离散系统稳定。当前算法中 alpha 都在 0 到 1 之间，因此这些 EWMA 子模块稳定。

## 4. 快慢基线特征提取

### 4.1 快速 EWMA

对每个车轮：

```text
fast_i[n] = fast_i[n-1] + fast_alpha * (w_i[n] - fast_i[n-1])
```

z 域：

```text
F_i(z) / W_i(z) = fast_alpha / (1 - (1 - fast_alpha) z^-1)
```

默认：

```text
fast_alpha = 0.45
```

快速 EWMA 对当前轮速变化响应较快。

### 4.2 慢速 EWMA

对每个车轮：

```text
slow_i[n] = slow_i[n-1] + slow_alpha * (w_i[n] - slow_i[n-1])
```

z 域：

```text
S_i(z) / W_i(z) = slow_alpha / (1 - (1 - slow_alpha) z^-1)
```

默认：

```text
slow_alpha = 0.001046674642901591
```

慢速 EWMA 作为长期基线。它不是每帧都更新，只有满足以下条件时才更新：

```text
speed_valid
and not alarm
and (recovery_active or score < freeze_enter)
```

因此，慢速基线本身是条件更新的。若只看单次更新公式，可以写成 z 域一阶 IIR；但包含更新冻结逻辑后，它不再是严格 LTI 系统。

### 4.3 恢复期慢速 EWMA

报警结束后进入 recovery 阶段时，慢速基线使用更大的 alpha：

```text
recovery_slow_alpha = 0.08
```

此时：

```text
slow_i[n] = slow_i[n-1] + recovery_slow_alpha * (w_i[n] - slow_i[n-1])
```

目的：让慢速基线更快跟随新的正常状态，减少报警后的残留偏差。

## 5. 四轮差分归一化

对四轮值：

```text
[fl, fr, rl, rr]
```

先计算总和：

```text
total = fl + fr + rl + rr
norm = 4 / total
```

若 `total` 太小，则 `norm = 0`。

归一化差分：

```text
d0 = (fl - fr) * norm
d1 = (rl - rr) * norm
d2 = (rl - fl) * norm
d3 = (rr - fr) * norm
```

分别对 `fast` 和 `slow` 计算：

```text
fast_diffs[n] = diffs_wheels(fast[n])
slow_diffs[n] = diffs_wheels(slow[n])
```

注意：这里有除以四轮总和的归一化，因此严格来说这是非线性变换，不能整体写成线性 z 域传递函数。

## 6. 残差组合 raw

先计算：

```text
residual_i[n] = slow_diffs_i[n] - fast_diffs_i[n]
```

然后组合：

```text
raw[n] = 0.5 * (residual_0[n] + residual_3[n]
                - residual_2[n] - residual_1[n])
```

这个组合用于突出特定四轮差分模式下的异常。

如果忽略 `diffs_wheels()` 中的归一化非线性，把 `fast_diffs` 和 `slow_diffs` 视为已知输入，则 `raw[n]` 是线性加权和。

## 7. 残差平滑与 legacy_combind

残差 `raw[n]` 再经过 EWMA 平滑：

```text
filtered[n] = filtered[n-1] + filter_alpha * (raw[n] - filtered[n-1])
```

z 域：

```text
Filtered(z) / Raw(z)
  = filter_alpha / (1 - (1 - filter_alpha) z^-1)
```

默认：

```text
filter_alpha = 0.65
```

输出缩放：

```text
legacy_combind[n] = filtered[n] * output_scale
```

默认：

```text
output_scale = 200.0
```

## 8. 单轮相对偏差 wheel_feature

对每个车轮，计算其相对其他三个车轮均值的偏差：

```text
ref_i[n] = average(w_j[n]), j != i
rel_i[n] = w_i[n] / ref_i[n] - 1
```

分别计算：

```text
fast_rel_i[n]
slow_rel_i[n]
```

单轮残差：

```text
wheel_residual_i[n] = (fast_rel_i[n] - slow_rel_i[n]) * 100
```

取绝对值最大的车轮：

```text
wheel_feature[n] = wheel_residual_k[n]
k = argmax_i abs(wheel_residual_i[n])
```

由于包含除法、绝对值和 `argmax`，该分支不能用线性 z 域传递函数表示。

默认配置：

```text
use_wheel_feature = False
```

因此默认检测特征为：

```text
combind[n] = legacy_combind[n]
```

如果开启 `use_wheel_feature`：

```text
combind[n] =
  wheel_feature[n],  if abs(wheel_feature[n]) > abs(legacy_combind[n])
  legacy_combind[n], otherwise
```

## 9. 自适应中心线与 innovation

检测器维护中心线：

```text
center[n]
```

首次输入时：

```text
center[0] = combind[0]
innovation[0] = 0
```

之后：

```text
innovation[n] = combind[n] - center[n-1]
```

在允许自适应时：

```text
center[n] = center[n-1] + center_alpha * (combind[n] - center[n-1])
```

z 域子模块：

```text
Center(z) / Combind(z)
  = center_alpha / (1 - (1 - center_alpha) z^-1)
```

默认：

```text
center_alpha = 0.0435206345569064
```

但中心线只在：

```text
adapt_noise and not alarm
```

时更新，所以完整中心线逻辑是条件更新系统，不是严格 LTI 系统。

## 10. 自适应噪声估计

定义：

```text
abs_x[n] = abs(innovation[n])
```

在允许自适应时：

```text
noise[n] = noise[n-1] + noise_alpha * (abs_x[n] - noise[n-1])
noise[n] = max(noise_floor, noise[n])
```

默认：

```text
noise_alpha = 0.00402139496108789
noise_floor = 0.08
```

若忽略 `abs()`、`max()` 和条件更新，则 EWMA 子模块的 z 域形式是：

```text
Noise(z) / AbsInnovation(z)
  = noise_alpha / (1 - (1 - noise_alpha) z^-1)
```

完整噪声估计不是线性系统，因为包含：

- 绝对值
- 下限钳位
- 条件更新

## 11. 动态进入/退出阈值

进入阈值：

```text
enter_threshold[n] =
  max(enter_min, enter_noise_gain * noise[n])
```

退出阈值：

```text
exit_threshold[n] =
  max(exit_min, exit_noise_gain * noise[n])
```

默认：

```text
enter_min = 0.7476840122076729
enter_noise_gain = 3.140654072838856

exit_min = 0.1928536858667876
exit_noise_gain = 2.6050079637872177
```

这一步是非线性阈值生成，不能写成单个 z 域传递函数。

## 12. 泄漏证据累计

先对 `innovation` 限幅：

```text
evidence_x[n] =
  clamp(innovation[n], -evidence_input_cap, evidence_input_cap)
```

默认：

```text
evidence_input_cap = 1.10
```

正向证据：

```text
pos_evidence[n] =
  max(0,
      evidence_decay * pos_evidence[n-1]
      + evidence_x[n]
      - enter_threshold[n])
```

负向证据：

```text
neg_evidence[n] =
  max(0,
      evidence_decay * neg_evidence[n-1]
      - evidence_x[n]
      - enter_threshold[n])
```

默认：

```text
evidence_decay = 0.9709366109407416
```

当前证据：

```text
evidence[n] = max(pos_evidence[n], neg_evidence[n])
score[n] = evidence[n]
```

带符号证据：

```text
signed_evidence[n] =
  pos_evidence[n],  if pos_evidence[n] >= neg_evidence[n]
  -neg_evidence[n], otherwise
```

如果只看泄漏积分器的线性部分：

```text
e[n] = evidence_decay * e[n-1] + u[n]
```

其中：

```text
u[n] = evidence_x[n] - enter_threshold[n]
```

z 域为：

```text
E(z) / U(z) = 1 / (1 - evidence_decay z^-1)
```

但实际证据累计还包含：

- 输入限幅
- 正负方向分离
- 阈值扣除
- `max(0, ...)` 半波整流
- 动态阈值

因此完整证据累计是非线性的。

## 13. 报警状态机

报警逻辑分为未报警和已报警两种状态。

### 13.1 未报警状态

如果当前未报警：

```text
alarm[n] =
  evidence[n] >= evidence_on
  or abs(innovation[n]) >= instant_on
```

默认：

```text
evidence_on = 1.419128576581635
instant_on = 999.0
```

由于 `instant_on` 很大，默认基本依赖 `evidence_on` 触发。

### 13.2 已报警状态

如果上一帧已经报警：

```text
alarm[n] =
  evidence[n] > evidence_off
  or abs(innovation[n]) > exit_threshold[n]
```

默认：

```text
evidence_off = 0.2849447795174631
```

这形成报警滞回：

- 进入报警使用较高阈值 `evidence_on`。
- 退出报警使用较低阈值 `evidence_off` 和动态 `exit_threshold`。

报警状态机不能用 z 变换表示，因为它是布尔状态逻辑。

## 14. warmup 预热

算法启动后前若干帧不允许报警：

```text
if frame_count <= warmup_frames:
    clear_evidence()
    alarm = False
```

默认：

```text
warmup_frames = 3000
```

预热目的：

- 让 `fast`、`slow`、`center`、`noise` 等状态先稳定。
- 避免初始化阶段误报。

## 15. 低速处理

如果：

```text
avg_speed < min_avg_speed
```

则认为速度无效。

此时若上一帧已经报警，且：

```text
hold_alarm_below_min_speed = True
```

则继续保持报警：

```text
alarm = True
```

否则清空证据并关闭报警：

```text
clear_evidence()
alarm = False
```

默认：

```text
hold_alarm_below_min_speed = True
```

## 16. recovery 恢复逻辑

当上一帧报警、当前帧退出报警时，如果开启恢复：

```text
recovery_enabled = True
```

则启动两个计数器：

```text
recovery_frames_left = recovery_frames
recovery_holdoff_left = recovery_holdoff_frames
```

默认：

```text
recovery_frames = 100
recovery_holdoff_frames = 20
```

### 16.1 recovery_active

当：

```text
recovery_frames_left > 0
```

进入恢复期。

恢复期中：

- 慢速基线使用 `recovery_slow_alpha`，更快跟随。
- 检测器调用 `recover_towards()`，让 `center` 和 `noise` 更快接近当前特征。

恢复期中心线：

```text
center[n] =
  center[n-1] + recovery_center_alpha * (combind[n] - center[n-1])
```

恢复期噪声：

```text
noise[n] =
  noise[n-1] + recovery_noise_alpha * (abs(innovation[n]) - noise[n-1])
```

默认：

```text
recovery_center_alpha = 0.20
recovery_noise_alpha = 0.05
```

然后清空证据：

```text
clear_evidence()
```

### 16.2 recovery_holdoff

当：

```text
recovery_holdoff_left > 0
```

强制关闭报警并清空证据：

```text
clear_evidence()
alarm = False
```

目的：报警刚退出时给自适应基线和噪声估计一个短暂稳定窗口，避免马上重复进入报警。

## 17. z 变换能表示什么

可以用 z 变换表示的线性子模块：

```text
fast EWMA:
F_i(z) / W_i(z)
  = fast_alpha / (1 - (1 - fast_alpha) z^-1)

slow EWMA:
S_i(z) / W_i(z)
  = slow_alpha / (1 - (1 - slow_alpha) z^-1)

filtered residual:
Filtered(z) / Raw(z)
  = filter_alpha / (1 - (1 - filter_alpha) z^-1)

center tracking:
Center(z) / Combind(z)
  = center_alpha / (1 - (1 - center_alpha) z^-1)

noise EWMA linear core:
Noise(z) / AbsInnovation(z)
  = noise_alpha / (1 - (1 - noise_alpha) z^-1)

linear core of evidence accumulation:
E(z) / U(z)
  = 1 / (1 - evidence_decay z^-1)
```

不能用单个 z 域传递函数完整表示的部分：

```text
diffs_wheels() 中的总和归一化
wheel_relative_deviation() 中的除法
abs()
max()
min()
clamp()
argmax()
阈值比较
alarm on/off 状态机
warmup 逻辑
低速保持/清零逻辑
recovery 计数器和 holdoff 逻辑
条件更新 adapt_slow / adapt_noise
```

因此，最严谨的系统表达是：

```text
alarm_detection =
  z 域可描述的一阶 IIR/EWMA 特征提取
  + 非线性归一化和特征选择
  + 自适应阈值
  + 泄漏证据累计
  + 报警状态机
```

## 18. 简化数学模型

如果用于论文、报告或算法说明，可以用如下简化模型表达。

### 18.1 特征提取

```text
fast_i[n] = (1 - a_f) fast_i[n-1] + a_f w_i[n]
slow_i[n] = (1 - a_s) slow_i[n-1] + a_s w_i[n]

d_fast[n] = D(fast[n])
d_slow[n] = D(slow[n])

r[n] = C * (d_slow[n] - d_fast[n])

filtered[n] = (1 - a_p) filtered[n-1] + a_p r[n]

x[n] = output_scale * filtered[n]
```

其中：

```text
D(.) 是四轮归一化差分函数
C = 0.5 * [1, -1, -1, 1]
a_f = fast_alpha
a_s = slow_alpha
a_p = filter_alpha
x[n] = combind[n]
```

### 18.2 检测器

```text
v[n] = x[n] - c[n-1]

c[n] = c[n-1] + a_c * (x[n] - c[n-1])
sigma[n] = max(noise_floor,
               sigma[n-1] + a_n * (abs(v[n]) - sigma[n-1]))

T_enter[n] = max(enter_min, enter_noise_gain * sigma[n])
T_exit[n] = max(exit_min, exit_noise_gain * sigma[n])

u[n] = clamp(v[n], -cap, cap)

e_pos[n] = max(0, decay * e_pos[n-1] + u[n] - T_enter[n])
e_neg[n] = max(0, decay * e_neg[n-1] - u[n] - T_enter[n])
e[n] = max(e_pos[n], e_neg[n])
```

### 18.3 报警逻辑

```text
if alarm[n-1] == False:
    alarm[n] = e[n] >= evidence_on or abs(v[n]) >= instant_on
else:
    alarm[n] = e[n] > evidence_off or abs(v[n]) > T_exit[n]
```

再叠加：

```text
warmup
speed_valid
hold_alarm_below_min_speed
recovery
recovery_holdoff
```

得到最终报警输出。

## 19. 关键参数默认值

```text
min_avg_speed = 20.0
fast_alpha = 0.45
slow_alpha = 0.001046674642901591
filter_alpha = 0.65
output_scale = 200.0
use_wheel_feature = False
warmup_frames = 3000

noise_alpha = 0.00402139496108789
center_alpha = 0.0435206345569064
noise_floor = 0.08
enter_min = 0.7476840122076729
enter_noise_gain = 3.140654072838856
exit_min = 0.1928536858667876
exit_noise_gain = 2.6050079637872177
evidence_decay = 0.9709366109407416
evidence_on = 1.419128576581635
evidence_off = 0.2849447795174631
evidence_input_cap = 1.10
instant_on = 999.0
freeze_enter = 0.38225805635222615

recovery_enabled = True
recovery_frames = 100
recovery_holdoff_frames = 20
recovery_slow_alpha = 0.08
recovery_center_alpha = 0.20
recovery_noise_alpha = 0.05
hold_alarm_below_min_speed = True
```

## 20. 命令行用法

默认执行：

```bash
python wheel_cog_outputs/alarm_detection_local_fast.py
```

指定输入、输出和序列：

```bash
python wheel_cog_outputs/alarm_detection_local_fast.py \
  --input wheel_cog_outputs/wheel_cog_outputs/wheel_speed_raw_vs_corrected.csv \
  --output wheel_cog_outputs/wheel_cog_outputs/alarm_detection_results_fast.csv \
  --summary wheel_cog_outputs/wheel_cog_outputs/alarm_detection_summary_fast.json \
  --series corrected \
  --min-avg-speed 20.0
```

输出包括：

- 每帧检测结果 CSV。
- 汇总 JSON，包括总帧数、报警帧数、报警段数、首末报警时间和参数配置。

