# YOLO Detect 自迭代说明

本项目的 `detection` 分支用于 YOLO 目标检测模型的训练、训练后验收、模型 PK、单模型推理和人工修标后复核。

固定使用 yolo 环境：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python
```

目标检测评价口径和 pose 不同：

- `task: detect`
- `metric: bbox_iou_f1`
- 匹配规则：类别一致，并且 bbox IoU 大于等于 `match_iou`
- 当前默认 `match_iou: 0.50`

## 配置文件

当前有两个配置：

```text
configs/project.yaml
configs/detection.yaml
```

推荐先使用单数据集闭环配置：

```text
configs/detection.yaml
```

`configs/project.yaml` 保留了更通用的入口；`configs/detection.yaml` 只覆盖当前这套目标检测闭环。

这份配置和当前检测流程保持一致：

- `train.project` 指向本轮检测训练输出目录。
- `candidate_model` 默认留空。为空时，验收和 PK 会自动使用 `train.project/train.name/weights/best.pt`。
- `champion_model` 指向预训练 Champion 模型。

当前使用的数据集：

```text
/data/liuyuhao/junction_detection/junction_v1/v9_xuxian_yinying/data.yaml
```

对应模型路径：

```text
Candidate:
/data-ssd/libo/ultralytics/runs/detect/train_resule/junction_v17_xuxian_yinying/weights/best.pt

Champion:
/data/liuyuhao/junction_detection/runs/all_junctions/junction_v16_v9_filtered/weights/best.pt
```

## 完整闭环

### 1. 训练模型

目标检测训练入口统一使用：

```text
scripts/train.py
```

推荐命令：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/train.py \
  --data /data/liuyuhao/junction_detection/junction_v1/v9_xuxian_yinying/data.yaml \
  --model /data/liuyuhao/junction_detection/runs/all_junctions/junction_v16_v9_filtered/weights/best.pt \
  --project /data-ssd/libo/ultralytics/runs/detect/train_resule \
  --name junction_v17_xuxian_yinying \
  --epochs 20 \
  --device 5 \
  --imgsz 1280 \
  --batch 4
```

训练输出：

```text
/data-ssd/libo/ultralytics/runs/detect/train_resule/junction_v17_xuxian_yinying/
```

重点模型文件：

```text
/data-ssd/libo/ultralytics/runs/detect/train_resule/junction_v17_xuxian_yinying/weights/best.pt
/data-ssd/libo/ultralytics/runs/detect/train_resule/junction_v17_xuxian_yinying/weights/last.pt
```

### 2. 训练后验收

训练后验收使用人工 GT 作为评价基准，输出 Candidate 和 Champion 各自相对 GT 的 TP、FP、FN、Precision、Recall、F1。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_acceptance.py \
  --config configs/detection.yaml \
  --profile full
```

输出目录类似：

```text
runs/junction_v17_detection_acceptance_YYYYMMDD_HHMMSS/
```

本轮已验证输出：

```text
/data-ssd/libo/p100/atuo_iter/runs/junction_v17_detection_acceptance_20260707_163922/
```

重点查看：

```text
runs/<run_id>/metrics/overall_summary.csv
runs/<run_id>/metrics/*_per_image.csv
runs/<run_id>/visualizations/
runs/<run_id>/manifests/
runs/<run_id>/logs/run.log
```

本轮验收结果：

```text
candidate: precision=0.9715616046, recall=0.9667830922, f1=0.9691664583
champion:  precision=0.9593879997, recall=0.9699194526, f1=0.9646249823
```

### 2.1 单模型图片推理

当前检测推理可以直接使用 Ultralytics `YOLO.predict`。下面命令会对 val 图片目录批量推理，保存可视化图片和 YOLO txt 预测结果。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python - <<'PY'
from ultralytics import YOLO

model = YOLO("/data-ssd/libo/ultralytics/runs/detect/train_resule/junction_v17_xuxian_yinying/weights/best.pt")
model.predict(
    source="/data/liuyuhao/junction_detection/junction_v1/v9_xuxian_yinying/val/images",
    project="/data-ssd/libo/p100/atuo_iter/runs",
    name="junction_v17_detection_predict",
    device="5",
    imgsz=1280,
    conf=0.25,
    iou=0.45,
    batch=4,
    half=True,
    save=True,
    save_txt=True,
    save_conf=True,
)
PY
```

输出目录类似：

```text
runs/junction_v17_detection_predict/
```

本轮已验证输出：

```text
/data-ssd/libo/p100/atuo_iter/runs/junction_v17_detection_predict_20260707_164314/
```

重点查看：

```text
runs/<predict_run>/labels/
runs/<predict_run>/*.jpg
```

其中 `labels/` 是 YOLO detect txt 预测结果，可视化图片由 Ultralytics 保存到输出目录。

### 2.2 模型 PK

模型 PK 用于对比 Candidate 和 Champion，不读取人工 GT 作为评价基准，而是把 `champion_model` 的预测当作 pseudo-GT，统计 Candidate 相对 Champion 的 FP/FN。推荐把 PK 用在单张图片或图片目录上，快速确认两个模型在指定样本上的差异；整套数据集验收使用 `acceptance_profiles`。

`model_pk_profiles` 里推荐使用 `images` 指定单张图片或一级图片目录：

```yaml
model_pk_profiles:
  default:
    run_id: detection_model_pk
    output_root: runs
    candidate_model: ""
    champion_model: ""
    eval_datasets:
      - name: junction_debug
        images: /path/to/image_or_images_dir
        names:
          0: junction
```

`names` 可选；不配置时会读取 `candidate_model` 和 `champion_model` 的 YOLO `names`，按类别名把两个模型的预测映射到同一个类别空间后再判断是否匹配。这样两个模型类别 id 顺序不同，但类别名相同时仍会算作同类；类别名不同则不匹配。兼容旧用法：也可以继续使用 `data: /path/to/data.yaml` 和 `splits: [val]`，但这更接近完整验收场景。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_model_pk.py \
  --config configs/detection.yaml \
  --profile default
```

输出目录类似：

```text
runs/junction_v17_detection_model_pk_YYYYMMDD_HHMMSS/
```

本轮已验证输出：

```text
/data-ssd/libo/p100/atuo_iter/runs/junction_v17_detection_model_pk_20260707_164059/
```

重点查看：

```text
runs/<run_id>/metrics/overall_summary.csv
runs/<run_id>/metrics/model_pk_<name>_images_summary.json
runs/<run_id>/metrics/model_pk_<name>_images_per_image.csv
runs/<run_id>/visualizations/<name>/images/fp/
runs/<run_id>/visualizations/<name>/images/fn/
runs/<run_id>/cache/candidate/<name>/images/_pred_labels/
runs/<run_id>/cache/champion/<name>/images/_pred_labels/
```

本轮 PK 结果：

```text
precision_vs_champion=0.9851719198
recall_vs_champion=0.9696820137
f1_vs_champion=0.9773655971
```

指标解释：

- `precision_vs_champion`：Candidate 预测中有多少能匹配 Champion 预测。
- `recall_vs_champion`：Champion 预测中有多少被 Candidate 找到。
- `fp/`：Candidate 比 Champion 多出来、未匹配的预测样本。
- `fn/`：Champion 有预测但 Candidate 没匹配上的样本。

注意：模型 PK 的 FP/FN 是相对 Champion 的差异，不代表相对人工 GT 的绝对正确或错误。正式验收仍然使用 `scripts/run_acceptance.py`。

## 可视化颜色

验收和模型 PK 的 compare 图使用同一套颜色：

- 绿色：TP，成功匹配。
- 红色：FP，Candidate 多预测或未匹配预测。
- 橙色：FN，GT/Champion 有目标但 Candidate 没匹配上。
- 检测 compare 图标签：gt 面板只显示类别；champion/candidate 面板显示类别和置信度分数。

模型 PK 里的 `gt` 面板不是人工 GT，而是 Champion 预测生成的 pseudo-GT。

`fp/` 和 `fn/` 文件夹表示这张图属于哪类问题样本。为了方便人工判断，compare 图里仍会同时画出同一张图上的 TP、FP、FN。

## 人工修标

训练后验收使用人工 GT，所以人工修标应优先查看 acceptance 结果：

```text
runs/<acceptance_run>/visualizations/junction_v9_xuxian_yinying/val/fp/
runs/<acceptance_run>/visualizations/junction_v9_xuxian_yinying/val/fn/
```

确认是标签问题后，直接修正原始数据集里的 label 文件。

注意：修标后不要删除 acceptance run 目录，因为复核需要复用其中的 `_pred_labels` 预测缓存。

## 修标后复核

复核只重新读取修正后的 GT label，并复用原始预测缓存，不重新推理。

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/run_review_recheck.py \
  --run runs/<acceptance_run> \
  --config configs/detection.yaml \
  --profile full \
  --review-name review_after_label_fix
```

复核输出在：

```text
runs/<acceptance_run>/review/review_after_label_fix_YYYYMMDD_HHMMSS/
```

重点查看：

```text
runs/<acceptance_run>/review/review_after_label_fix_*/metrics/overall_summary.csv
runs/<acceptance_run>/review/review_after_label_fix_*/metrics/*_per_image.csv
runs/<acceptance_run>/review/review_after_label_fix_*/manifests/
```

## 下一轮迭代

如果修标后指标正常，可以进入下一轮训练：

```bash
/data1/libo/miniconda3/envs/yolo/bin/python scripts/train.py \
  --data /data/liuyuhao/junction_detection/junction_v1/v9_xuxian_yinying/data.yaml \
  --model /data-ssd/libo/ultralytics/runs/detect/train_resule/junction_v17_xuxian_yinying/weights/best.pt \
  --project /data-ssd/libo/ultralytics/runs/detect/train_resule \
  --name junction_v18_xuxian_yinying \
  --epochs 20 \
  --device 5 \
  --imgsz 1280 \
  --batch 4
```

然后重复：

```text
训练 -> 训练后验收 -> 人工修标 -> 修标后复核 -> 下一轮训练
```

## 测试

```bash
/data1/libo/miniconda3/envs/yolo/bin/python -m unittest discover -s tests -v
```
