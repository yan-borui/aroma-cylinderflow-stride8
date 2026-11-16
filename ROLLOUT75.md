**完整 Test75 / Validation75 评分：**使用 [EVALUATE75.md](EVALUATE75.md) 的 `evaluate75` 入口。本页 `rollout75` 保留 Test65 前缀评分兼容入口。

# AROMA 原生 75 帧预测

从真实第 0 帧开始，连续预测未来 74 帧，输出总计 75 帧 UVP。CylinderFlow dt=0.08，原始帧索引 0,8,…,592，覆盖 5.92 秒。固定已有选中权重，保留当前 `cylinderflow-stride8` 分支的 Train75、归一化和模型依赖。

初始物理场只编码一次，latent 连续递推 74 次，每步保留原生 4 步 DDPM/v-prediction，最后解码全部 latent。窗口重启、物理场重编码和重叠融合均未引入。保持原采样标签 0/1/2，三条完整轨迹各自递推、分别评分。

## 运行

在匹配权重的既有生产 CUDA 环境中、仓库根目录运行。默认单 GPU、FP32、完整 Validation100，输出目录必须不存在。

```bash
python -m cylinderflow.rollout75 \
  --dataset /absolute/path/to/data.h5 \
  --manifest /absolute/path/to/manifest.json \
  --prepared /absolute/path/to/matching_prepared \
  --checkpoint /absolute/path/to/selected_dynamics.pt \
  --ae-checkpoint /absolute/path/to/matching_ae.pt \\
  --scope validation --device cuda:0 \
  --output-dir /absolute/path/to/new_rollout75
```

`--config` 默认读取本仓库 `cylinderflow_config.json`，校验后使用所选 checkpoint 中保存的配置。AE 必须匹配 dynamics checkpoint 内的 AE 身份。 可通过 `--case-count 3` 取 split 顺序前三条，或 `--indices 1000 1001 1002` 指定 Validation 全局编号；全部请求及失败均保留。

Test 使用 [TEST.md](TEST.md) 的官方 Test65 转换目录，加 `--scope test --test-data-dir /absolute/path/to/test65`。Test 编号为 0–99。同样生成 75 帧，但 Test65 只有前 65 帧参考，因此仅评分未来 1–64 帧，最后 10 帧仅保存预测。

## 递推、指标与输出

原 `rollout` 默认仍为 64 次递推；本入口显式设置 `future_frames=74`。原训练、选权、65 帧评测入口和归档格式保持兼容。

Validation 完成全部生成后才读取 75 帧参考。分别报告 `prefix64`（未来 1–64 帧）与 `full74`（未来 1–74 帧）指标，二者单独汇总。UV RMSE 分母与时间频谱轴按实际预测长度计算；整体面积加权 UV relative RMSE 与逐帧 UV relative RMSE 均保留。先对同轨迹三个采样的指标取平均，再对轨迹等权平均；不将物理场均值当成原生三样本评分。 失败样本显式计数，存在失败时非零退出。

- `identity.json`、`request.json`、`source.zip`：数据/权重/AE/统计与配置、生成和评分范围、采样种子规则、实际运行环境及源码。
- `trajectory_*_seed_*/prediction.npz`：物理预测与边界回填前场，均为 `[75,N,3]`；另含网格、标签、原始帧号、时间、采样种子及原生诊断。使用独立 `cylinderflow.native_rollout75.v1` 格式。
- `reference.npz`：生成后读取的真实参考（Validation75 或 Test65），独立保存时间轴。
- `metrics.json`、根目录 `summary.json`：逐样本记录及按轨迹汇总的不同预测长度指标。
- `failure.json`：初始化或逐样本失败；已完成预测和指标保留。

2026-11-16：静态语法、名称、差异及原生递推接口核查完成。未核实生产等价 CUDA 环境，未执行 GPU 推理或替代环境软件测试；本轮交付代码和命令，不含运行结果。
