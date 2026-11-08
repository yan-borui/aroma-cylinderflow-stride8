# AROMA on MegaFlow2D Ellipse

This branch implements the original high-resolution mesh UVP task: file frames 1,16,…,961, one observation and 64 predictions, Train782 / Validation100 / Test100. All training uses four GPUs and checkpoint selection uses full Validation100.

Start with [ELLIPSE.md](ELLIPSE.md) for data preparation, training, recovery, budgets and verification status. Use [TEST.md](TEST.md) for final evaluation. Architecture and optimizer settings are inherited from the Airfoil four-GPU source; [AIRFOIL.md](AIRFOIL.md) is historical reference, and its launch commands do not define this branch’s task.

```bash
export RAW_DATA_DIR=/data/datasets/megaflow2d
export DATA_DIR=/data/datasets/megaflow2d_ellipse_stride15
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_ROOT=/data/runs/ellipse_aroma_seed0
bash scripts/ellipse_4gpu.sh train
```

## AE → dynamics handoff

The wrapper trains AE, selects `ae/best.pt` on full Validation100, caches Train65 posterior means, then trains dynamics. PREPARED_DIR defaults to `$DATA_DIR/prepared_aroma_4gpu`; the run retains its statistics/cache in `$RESULT_ROOT/prepared`. Final evaluation defaults to `dynamics/best.pt` and its matching `ae/best.pt`; CHECKPOINT and AE_CHECKPOINT can explicitly identify the selected pair. Checkpoint/cache dependencies must match.
