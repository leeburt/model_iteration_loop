# YOLO Pose 自迭代 MVP

本项目用于 YOLO pose 单类别单关键点端口识别的训练、验收和人工修标后复核。

固定使用 yolo 环境：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python
```

## 配置文件

当前有两个配置：

```text
configs/project.yaml
configs/merge_config_yolo26_0609.yaml
```

推荐先使用单数据集闭环配置：

```text
configs/merge_config_yolo26_0609.yaml
```

它只使用：

```text
/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_yolo26_0609.yaml
```

配置里的 `candidate_model` 默认留空。为空时，验收会自动使用：

```text
train.project/train.name/weights/best.pt
```

## 完整闭环

### 1. 训练模型

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/train_pose.py \
  --config configs/merge_config_yolo26_0609.yaml
```

训练输出默认在配置里的 `train.project/train.name` 下，例如：

```text
/data-ssd/libo/p100/atuo_iter/train_result/external_port/keypoint_yolo11x_pose_merge_config_yolo26_0609/
```

验收会默认读取：

```text
/data-ssd/libo/p100/atuo_iter/train_result/external_port/keypoint_yolo11x_pose_merge_config_yolo26_0609/weights/best.pt
```

### 2. 训练后验收

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_acceptance.py \
  --config configs/merge_config_yolo26_0609.yaml \
  --profile full
```

验收会评估配置里的：

```text
merge_config_yolo26_0609/train
merge_config_yolo26_0609/val
```

输出目录类似：

```text
runs/merge_config_yolo26_0609_acceptance_YYYYMMDD_HHMMSS/
```

重点查看：

```text
runs/<run_id>/metrics/overall_summary.csv
runs/<run_id>/metrics/*_per_image.csv
runs/<run_id>/diff/
runs/<run_id>/manifests/
runs/<run_id>/logs/run.log
```

其中：

- `overall_summary.csv` 是总体 TP/FP/FN、Precision、Recall、F1。
- `*_per_image.csv` 是逐图结果。
- `diff/` 保存 FP/FN 图片、GT label、预测 label，供人工检查。
- `manifests/` 记录数据版本、hash 和数据泄漏检查结果。

### 3. 人工修标

根据第 2 步的 `diff/` 结果，人工检查 FP/FN：

```text
runs/<run_id>/diff/candidate/merge_config_yolo26_0609/train/
runs/<run_id>/diff/candidate/merge_config_yolo26_0609/val/
```

确认是标签问题后，直接修正原始数据集里的 label 文件。

注意：修标后不要删除第 2 步的 run 目录，因为复核需要复用其中的 `_pred_labels` 预测缓存。

### 4. 修标后复核

复核只重新读取修正后的 GT label，并复用原始预测缓存，不重新推理。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_review_recheck.py \
  --run runs/<run_id> \
  --config configs/merge_config_yolo26_0609.yaml \
  --profile full \
  --review-name review_after_label_fix
```

这里的 `<run_id>` 替换为第 2 步生成的目录名，例如：

```text
runs/merge_config_yolo26_0609_acceptance_20260703_153000
```

复核输出在：

```text
runs/<run_id>/review/review_after_label_fix_YYYYMMDD_HHMMSS/
```

重点查看：

```text
runs/<run_id>/review/review_after_label_fix_*/metrics/overall_summary.csv
runs/<run_id>/review/review_after_label_fix_*/metrics/*_per_image.csv
runs/<run_id>/review/review_after_label_fix_*/manifests/
```

### 5. 下一轮迭代

如果修标后指标正常，可以进入下一轮训练：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/train_pose.py \
  --config configs/merge_config_yolo26_0609.yaml
```

然后重复“训练后验收 -> 人工修标 -> 修标后复核”。

## 通用配置

`configs/project.yaml` 是通用配置，保留了额外 benchmark 数据集。运行方式相同：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_acceptance.py \
  --config configs/project.yaml \
  --profile full
```

## 测试

```bash
/data1/libo/miniconda3/envs/yolo/bin/python -m unittest discover -s tests -v
```
