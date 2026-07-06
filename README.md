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
runs/<run_id>/visualizations/
runs/<run_id>/manifests/
runs/<run_id>/logs/run.log
```

其中：

- `overall_summary.csv` 是总体 TP/FP/FN、Precision、Recall、F1。
- `*_per_image.csv` 是逐图结果。
- `visualizations/` 按数据集和 split 保存 Candidate FP/FN、新增错误和改进样本。
- `manifests/` 记录数据版本、hash 和数据泄漏检查结果。

### 2.1 单模型图片推理

指定一个模型和一张图片，快速生成预测标签和可视化：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/predict_pose.py \
  --model /data-ssd/libo/p100/atuo_iter/train_result/external_port/keypoint_yolo11x_pose_merge_config_yolo26_0609_test_continus/weights/best.pt \
  --source /path/to/image.png \
  --output runs/predict_one_image
```

也可以把 `--source` 指向图片目录，脚本会批量推理该目录下的图片：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/predict_pose.py \
  --model /path/to/best.pt \
  --source /path/to/images_dir \
  --output runs/predict_images_dir \
  --device 0 \
  --imgsz 1536 \
  --batch 4 \
  --conf 0.25 \
  --iou 0.45
```

已验证可执行示例：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/predict_pose.py \
  --model /data-ssd/libo/p100/atuo_iter/train_result/external_port/keypoint_yolo11x_pose_merge_config_yolo26_0609_test_continus/weights/best.pt \
  --source /data-ssd/libo/p100/atuo_iter/runs/merge_config_yolo26_0609_acceptance_20260706_135712/diff/candidate/merge_config_yolo26_0609/val/images \
  --output runs/predict_diff_val_best_pt_test \
  --device 0 \
  --imgsz 1536 \
  --batch 4 \
  --conf 0.25 \
  --iou 0.45
```

如果不指定 `--output`，默认输出到：

```text
runs/predict_pose_YYYYMMDD_HHMMSS/
```

重点查看：

```text
runs/<predict_run>/labels/
runs/<predict_run>/visualizations/
runs/<predict_run>/summary.csv
runs/<predict_run>/summary.json
runs/<predict_run>/logs/run.log
```

其中 `labels/` 是 YOLO pose txt 预测结果，`visualizations/` 是预测框和关键点覆盖图，`summary.csv/json` 记录每张图片的预测数量和输出文件路径。

常用参数：

- `--source` 支持单张图片或一级图片目录。
- `--batch` 控制批量推理 batch size；只有 `--source` 是图片目录时才有明显意义，显存不足时调小。
- `--no-visuals` 只保存 txt 和 summary，不保存可视化图。
- `--no-half` 关闭 FP16 推理。
- `--no-progress` 关闭进度条。

### 2.2 模型 PK

模型 PK 用于批量对比两个模型，不读取人工 GT 作为评价基准，而是把 `champion_model` 的预测当作 pseudo-GT，统计 `candidate_model` 相对 Champion 的 FP/FN。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_model_pk.py \
  --config configs/merge_config_yolo26_0609.yaml \
  --profile default
```

当前默认配置在：

```yaml
model_pk_profiles:
  default:
    run_id: merge_config_yolo26_0609_model_pk
    output_root: runs
    candidate_model: ""
    champion_model: ""
    eval_datasets:
      - name: paper_benchmark_100
        data: /data-ssd/libo/p100/yolo_utils/dataset/external_inline/paper_benchmark_100/data.yaml
        splits: [val]
```

字段含义：

- `candidate_model` 为空时，自动使用 `train.project/train.name/weights/best.pt`。
- `champion_model` 为空时，继承顶层 `models.champion_model`；模型 PK 必须有 Champion。
- `eval_datasets` 可以配置多个数据集；每个数据集可以配置多个 split。

输出目录类似：

```text
runs/merge_config_yolo26_0609_model_pk_YYYYMMDD_HHMMSS/
```

重点查看：

```text
runs/<run_id>/metrics/overall_summary.csv
runs/<run_id>/metrics/model_pk_<dataset>_<split>_summary.json
runs/<run_id>/metrics/model_pk_<dataset>_<split>_per_image.csv
runs/<run_id>/visualizations/<dataset>/<split>/fp/
runs/<run_id>/visualizations/<dataset>/<split>/fn/
runs/<run_id>/cache/candidate/<dataset>/<split>/_pred_labels/
runs/<run_id>/cache/champion/<dataset>/<split>/_pred_labels/
```

指标解释：

- `precision_vs_champion`：Candidate 预测中有多少能匹配 Champion 预测。
- `recall_vs_champion`：Champion 预测中有多少被 Candidate 找到。
- `fp/`：Candidate 比 Champion 多出来、未匹配的预测。
- `fn/`：Champion 有预测但 Candidate 没匹配上的样本。

注意：模型 PK 的 FP/FN 是相对 Champion 的差异，不代表相对人工 GT 的绝对正确或错误。正式验收仍然使用 `scripts/run_acceptance.py`。

### 3. 人工修标

根据第 2 步的 `visualizations/` 结果，人工检查 FP/FN：

```text
runs/<run_id>/visualizations/merge_config_yolo26_0609/train/fp/
runs/<run_id>/visualizations/merge_config_yolo26_0609/train/fn/
runs/<run_id>/visualizations/merge_config_yolo26_0609/val/fp/
runs/<run_id>/visualizations/merge_config_yolo26_0609/val/fn/
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
