# YOLO Pose 自迭代 MVP

本项目用于 YOLO pose 单类别单关键点端口识别的训练、验收和人工修标后复核。

固定使用 yolo 环境：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python
```

## 训练

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/train_pose.py \
  --config configs/project.yaml
```

## 验收

默认完整配置：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_acceptance.py \
  --config configs/project.yaml \
  --profile full
```

## 人工修标后复核

复核会复用原始 `_pred_labels` 预测缓存，不重新推理。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_review_recheck.py \
  --run runs/<run_id> \
  --config configs/project.yaml \
  --profile full \
  --review-name review_after_label_fix
```

## 测试

```bash
/data1/libo/miniconda3/envs/yolo/bin/python -m unittest discover -s tests -v
```
