# 前65帧动力学训练

CylinderFlow 的 AROMA 使用 stored frames 0–64 训练，覆盖前 64 个时间间隔。
AROMA 每条轨迹遍历 64 个相邻帧对，即 0→1 至 63→64。
原 75 帧 HDF5、Train75 归一化、模型、优化器、学习率规则和 Validation 选优规则沿用既有配方。
表示学习保留原 75 帧范围；本入口复用既有选优 AE 和对应的 posterior-mean Train75 latent 缓存，从头训练动力学。

四卡配置每卡 batch32、accumulation1，全局 batch128。
5,000 轮；每轮 64,000 个相邻帧对、512 次更新；全程 320,000,000 个样本、2,560,000 次更新。轨迹分组保留末组 104 条，其余七组各 128 条。
学习率时钟继续沿用原训练配方。
评价从 frame0 预测未来 64 帧。时间间隔为 0.08，沿用原边界处理。
Test 保持封存。

## 训练与恢复

复用已经安装的四卡训练环境，以及该数据集、该方法已有的数据准备目录。
将 PREPARED 指向原 AROMA 运行内含 train_latents.h5 的 prepared 目录，并使用生成该缓存的同一份选优 AE。
路径使用绝对路径。RESULT_ROOT 使用新的独立目录。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA=/absolute/path/to/cylinderflow_stride8_75frames.h5
export MANIFEST=/absolute/path/to/matching_manifest.json
export PREPARED=/absolute/path/to/original_aroma_run/prepared
export AE_CHECKPOINT=/absolute/path/to/original_aroma_run/ae/best.pt
export RESULT_ROOT=/absolute/path/to/cylinderflow_aroma_prefix65
bash scripts/train_prefix65_4gpu.sh train
```

恢复同一次训练，保留上述变量并运行：

```bash
bash scripts/train_prefix65_4gpu.sh resume
```

恢复记录检查完整配置；仅恢复本次 prefix65 运行产生的 checkpoint。
日志与退出记录位于 RESULT_ROOT 下，模型阶段目录为 dynamics。

若选优 AE 已有而 latent 缓存尚未生成，先完成原方法的统计量准备，再运行：

```bash
python -m cylinderflow prepare --latents \
  --config cylinderflow_config_prefix65_4gpu.json \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --ae-checkpoint "$AE_CHECKPOINT" --device cuda:0 \
  --output-dir /absolute/path/to/new_cache_build_log
```

缓存记录 AE 身份和归一化，动力学入口会核对两者。新动力学运行使用独立目录并复用该缓存。

## 评价与回传

训练按原 Validation24 规则选优。完成后使用本次 best.pt 评价完整 Validation100：

```bash
python -m cylinderflow evaluate \
  --config cylinderflow_config_prefix65_4gpu.json \
  --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$RESULT_ROOT/prepared" \
  --checkpoint "$RESULT_ROOT/dynamics/best.pt" \
  --ae-checkpoint "$AE_CHECKPOINT" \
  --mode validation --device cuda:0 --output-dir "$RESULT_ROOT/validation100"
```

回传完整指标、训练曲线、选优记录、运行配置和退出记录。
现有物理指标、预测导出和重复采样入口可使用本次选中权重，配置指定本页的 prefix65 配置。

交付验证覆盖静态语法、配置、取样范围及 diff 检查。正式四卡环境及训练运行结果由接收端确认。
