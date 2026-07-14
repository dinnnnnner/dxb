# 爆胎事件数据增强

脚本读取人工确认后的 `event_time_labels.csv` 和批处理生成的四轮轮速 CSV，输出固定长度的事件、正常样本以及 `manifest.csv`。

默认事件样本范围是爆胎前 40 秒、爆胎后 10 秒。当前快速报警算法需要约 30 秒预热，因此不要把 `--pre-s` 调到 30 秒以下后再直接评价该算法。

## 小规模试运行

```bash
python wheel_cog_outputs/build_augmented_event_dataset.py \
  --output-dir /tmp/blowout_aug_smoke \
  --aug-per-event 2 \
  --normal-per-event 1
```

## 生成正式增强集

```bash
python wheel_cog_outputs/build_augmented_event_dataset.py \
  --aug-per-event 50 \
  --normal-per-event 10
```

默认输出到 `wheel_cog_outputs/augmented_event_dataset/`：

- `samples/*.csv`：可由现有快速报警算法读取的四轮轮速。
- `manifest.csv`：每个样本的来源、样本内事件时间和增强参数。
- `dataset_config.json`：本次生成使用的全部参数。

事件样本包含一份未增强基准样本（编号 `000`）和指定数量的增强样本。增强包括：

- 爆胎在样本内的位置随机变化 ±0.5 秒。
- 时间伸缩 0.95～1.05。
- 四轮共同速度缩放 0.95～1.05。
- 根据该文件爆胎前正常段估计四轮噪声标准差，叠加零均值高斯噪声，强度 0.10～0.50。
- 30% 概率模拟 1～3 个采样点缺失并线性插值。

时间变换后，脚本会把新的事件位置写入 `event_time_in_sample_s`，不需要手工换算。

## 数据隔离

训练和验证必须按照 `source_event_id` 分组。比如留 E01 测试时，所有 `source_event_id=E01` 的原始、增强和正常窗口都必须从训练集排除。增强样本不能用于最终事件召回率和误报率统计；最终指标只使用未增强的真实数据。

## 动态查看全部样本

先生成一次轻量的批量评价表，不预先生成单样本 HTML：

```bash
python wheel_cog_outputs/evaluate_augmented_fast_batch_display.py --html-mode none
```

再启动只监听本机的动态查看器：

```bash
python wheel_cog_outputs/serve_augmented_fast_batch_display.py
```

浏览器打开 `http://localhost:8765`。首页可搜索和筛选全部 488 个样本；点击 `sample_id` 时才读取对应 CSV、运行算法并生成交互图。查看器默认在内存中缓存最近 12 个页面，不会为 488 个样本写出静态 HTML。按 `Ctrl+C` 停止服务器。
