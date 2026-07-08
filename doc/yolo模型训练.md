# YOLO 模型训练与自迭代验收方案

## 1. 项目背景

本方案第一版只聚焦 YOLO 目标检测任务。关键点检测、实例分割可以沿用总体闭环，但匹配规则、错误定义和可视化规范需要单独扩展，不纳入第一版范围。

在 YOLO 目标检测模型训练中，不能只依据训练日志中的 `mAP`、`Precision`、`Recall` 判断模型是否可以上线。每轮训练完成后，需要系统性回答以下问题：

1. 新增训练数据是否被模型有效学习；
2. 历史训练数据是否出现拟合退化或遗忘；
3. 验证集上的泛化能力是否提升；
4. 新模型在固定 Benchmark 上是否真正优于当前线上模型；
5. 各数据集中的 FP（误检）和 FN（漏检）主要集中在哪些样本；
6. FP/FN 中是否存在漏标、错标、框偏移等数据质量问题；
7. 候选模型是否具备上线或灰度发布条件；
8. 人工确认的问题样本如何形成新数据版本并回流到下一轮迭代。

因此，需要建设一套“训练后自动验收与人工回流自迭代系统”，在每轮训练结束后自动完成数据版本冻结、多数据集评估、新旧模型对比、FP/FN 可视化、统一报告生成、人工审核结果记录和修标后复核报告生成。

第一版明确不做以下能力：

- 不做复杂审核任务队列、工单状态或权限系统；
- 不做主动学习自动选样；
- 不做线上采样平台；
- 不做强制上线阻断，只输出自动上线建议；
- 不自动重新训练，修标后是否进入下一轮训练由人工决定。

## 2. 建设目标

系统目标是将模型训练后的验收过程标准化、自动化、可追溯化。

整体闭环如下：

```Plain Text
flowchart LR
    A[YOLO data config] --> B[生成 dataset_manifest]
    B --> C[训练 Candidate]
    C --> D[生成 run_manifest]
    D --> E[Candidate/Champion 统一推理]
    E --> F[保存 prediction_cache]
    F --> G[多数据集指标评估]
    G --> H[Candidate 与 Champion 对比]
    H --> I[FP/FN 可视化]
    I --> J[生成原始验收报告]
    J --> K[人工查看 FP/FN]
    K --> L{是否存在标签问题}
    L -- 否 --> M[人工决定灰度或上线]
    L -- 是 --> N[修正标签并生成新 dataset_version]
    N --> O[复用 prediction_cache 重新匹配 GT]
    O --> P[生成独立复核报告]
    P --> Q[人工决定是否进入下一轮训练]
```

每轮训练交付物不应只是一个 `best.pt`，而应包括：

```Plain Text
模型权重
+ dataset_manifest
+ run_manifest
+ prediction_cache
+ 训练配置
+ 统一评估协议
+ 多数据集指标
+ Candidate/Champion 对比
+ FP/FN 可视化图集
+ 新模型新增错误样本
+ HTML/PDF/CSV/JSON 原始验收报告
+ 人工审核结果
+ 修标后新数据版本
+ 独立复核报告
```

## 3. 数据版本管理

数据版本管理是第一版必须建设的能力。没有数据版本，就无法判断某次训练、某份报告、某次人工复核到底基于哪些图片和标签。

### 3.1 版本粒度

第一版采用整包版本管理：

```Plain Text
dataset_version = 一次训练或评估使用的完整数据包版本
```

一个 `dataset_version` 内部记录所有 split 和来源，包括但不限于：

- `train`
- `val`
- `benchmark`
- `hard_case`
- 其他由 YOLO data config 声明的数据路径

YOLO data config 是训练与评估数据来源的事实入口。系统读取 YOLO data config 和目录结构后生成 `dataset_manifest`，不要求人工手写 manifest。

### 3.2 dataset_manifest

`dataset_manifest` 用于证明“当前数据版本包含哪些图片、标签、类别、split 和来源”。建议输出为 JSON 或 YAML。

核心字段：

```Plain Text
manifest_version
dataset_version
created_at
yolo_data_config_path
yolo_data_config_hash
class_names
class_count
image_count
object_count
split_summary
records[]
```

单条样本记录至少包含：

```Plain Text
image_path
label_path
image_hash
label_hash
split
source
width
height
object_count
```

其中 `image_hash` 和 `label_hash` 必须记录，用于发现图片或标签被静默修改。只记录路径是不够的。

### 3.3 run_manifest

`run_manifest` 用于证明“某次训练或评估是如何产生的”。它和 `dataset_manifest` 分开维护，但在报告中必须绑定展示。

核心字段：

```Plain Text
run_id
created_at
candidate_model_path
candidate_model_hash
champion_model_path
champion_model_hash
pretrained_model_path
pretrained_model_hash
train_config_path
eval_protocol_path
dataset_version
class_names
train_start_time
train_end_time
infer_device
code_version
```

`pretrained_model_path`、训练类别、数据数量、时间信息属于 `run_manifest` 或报告摘要的一部分；图片与标签清单属于 `dataset_manifest`。

### 3.4 数据泄漏风险

系统不强制阻断 YOLO config 中的路径配置，但必须检测以下风险：

- `benchmark` 样本出现在训练路径中；
- `hard_case` 样本出现在训练路径中；
- 同一图片 hash 同时存在于训练 split 和评估 split；
- 同一标签 hash 被修改但 dataset_version 未变化；
- 同一路径在不同 split 中重复出现。

第一版处理策略：

- 不阻断流程；
- 在报告中标记为高风险；
- 自动上线建议通常降级为 `MANUAL_REVIEW`；
- 如果 benchmark 泄漏严重，可按配置降级为 `REJECT`。

## 4. 核心设计原则

### 4.1 固定 Benchmark 作为上线决策依据

YOLO data config 是系统读取数据路径的事实入口。下表描述的是不同数据集在评估报告中的推荐语义，不覆盖项目自己的 config 组织方式。

不同数据集承担不同职责：

| 数据集 | 主要用途 | 推荐训练关系 | 是否用于上线建议 |
| --- | --- | --- | --- |
| `train` / `old_train` | 检查历史训练样本拟合及遗忘 | 是 | 否 |
| `new_train` | 检查新增数据拟合及标注问题 | 是 | 否 |
| `validation` | 检查常规泛化能力 | 建议不参与训练 | 部分 |
| `benchmark` | 固定基准集，新旧模型公平比较 | 建议不参与训练 | 是 |
| `hard_case_set` | 历史困难样本或线上问题样本 | 建议不参与训练 | 是 |

其中：

- `new_train` 和 `old_train` 的结果主要用于发现训练数据和标注问题；
- `validation` 用于观察一般泛化能力；
- `benchmark` 是判断 Candidate 是否优于 Champion 的核心依据；
- `hard_case_set` 用于防止历史关键问题重新出现。

固定 Benchmark 应长期冻结，不应随单轮结果随意修改。若业务需要更新 Benchmark，应生成新的 benchmark 版本，并保留旧版本报告以便横向比较。

### 4.2 Candidate 与 Champion 必须在同一协议下评估

新旧模型必须使用完全相同的推理与评估参数，包括但不限于：

- 输入尺寸；
- 置信度阈值；
- NMS IoU 阈值；
- 最大检测数量；
- GT 与预测框的匹配 IoU 阈值；
- 类别匹配规则；
- 推理设备与推理模式；
- 后处理逻辑；
- 数据预处理逻辑。

这些参数必须写入 `eval_protocol`，并在 `run_manifest` 和报告中展示。

### 4.3 不只比较 mAP，还要比较 FP、FN 和新旧模型差异

总体 mAP 提升不一定意味着模型可以上线。例如：

- 关键类别 Recall 下降；
- 小目标 FN 增加；
- Benchmark 上出现新的高置信度 FP；
- Champion 正确、Candidate 却失败的样本增加；
- 历史困难样本退化；
- 新模型推理延迟或显存占用明显增加。

因此，验收报告需要同时展示：

1. 总体指标；
2. 每类别指标；
3. 各数据集的 FP/FN 数量；
4. 各数据集的 FP/FN 可视化；
5. Candidate 相对 Champion 新增的 FP/FN；
6. Candidate 相对 Champion 改进的样本；
7. 数据泄漏风险；
8. 固定 Benchmark 上的综合结论。

## 5. 错误样本定义

第一版只定义目标检测错误样本。

默认采用“类别一致且 IoU 大于等于阈值”作为 TP 匹配条件：

```Plain Text
match_iou_threshold = 0.5
```

匹配规则：

1. 同一类别内按预测置信度从高到低排序；
2. 每个预测框只允许匹配一个 GT；
3. 每个 GT 只允许被一个预测框匹配；
4. 满足类别一致且 IoU 大于等于阈值的最高优先级匹配记为 TP；
5. 未匹配到 GT 的预测框记为 FP；
6. 未被任何预测框匹配的 GT 记为 FN；
7. 类别错误的预测需要同时体现为错误类别的 FP 和真实类别的 FN。

| 类型 | 定义 |
| --- | --- |
| TP | 预测类别正确，且预测框与 GT 的 IoU 大于等于阈值 |
| FP | 预测框无法匹配任意 GT，或预测类别错误，或重复检测同一 GT |
| FN | GT 无法匹配任意预测框，或被错误类别预测 |

评估时建议同时按目标尺寸分层统计，例如 `small`、`medium`、`large`，用于发现小目标漏检问题。

## 6. 训练后自动验收流程

### 6.1 输入

每轮训练完成后，验收模块输入如下：

```Plain Text
Candidate 模型路径
Champion 模型路径
YOLO data config
统一评估协议 eval_protocol
训练配置
预训练模型信息
模型版本信息
输出目录
```

数据集列表不需要人工重复维护，系统从 YOLO data config 读取并冻结到 `dataset_manifest`。

### 6.2 自动执行流程

```Plain Text
flowchart TD
    A[读取 YOLO data config] --> B[生成 dataset_manifest]
    B --> C[生成 run_manifest]
    C --> D[加载 Candidate 与 Champion]
    D --> E[加载统一评估协议]
    E --> F[对所有配置数据集运行推理]
    F --> G[保存 prediction_cache]
    G --> H[计算总体及分类别指标]
    H --> I[计算 TP/FP/FN]
    I --> J[比较 Candidate 与 Champion]
    J --> K[检测数据泄漏风险]
    K --> L[输出 FP/FN 和新旧模型差异图]
    L --> M[生成 HTML/PDF/CSV/JSON 原始报告]
    M --> N[输出自动上线建议]
```

### 6.3 prediction_cache

`prediction_cache` 必须保存 Candidate 与 Champion 的原始预测结果，用于人工修标后不重新推理地复核指标。

建议字段：

```Plain Text
run_id
model_role          # candidate / champion
model_path
dataset_version
eval_protocol_hash
image_path
image_hash
predictions[]
```

单个预测框至少包含：

```Plain Text
class_id
class_name
confidence
x1
y1
x2
y2
```

人工修标后的复核流程必须复用 `prediction_cache`，只用新的 GT 重新执行匹配和指标计算。

## 7. 指标输出要求

### 7.1 各数据集总体指标

对 YOLO data config 中声明的各数据集分别输出：

```Plain Text
图像数量
GT 目标数量
预测目标数量
TP 数量
FP 数量
FN 数量
Precision
Recall
F1-score
mAP@0.5
mAP@0.5:0.95
平均 IoU
平均预测置信度
每图平均 FP
每图平均 FN
推理延迟 P50/P95
```

报告中应生成如下总览表：

| 数据集 | 模型 | Precision | Recall | F1 | mAP50 | mAP50-95 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_train | Champion |  |  |  |  |  |  |  |  |
| new_train | Candidate |  |  |  |  |  |  |  |  |
| validation | Champion |  |  |  |  |  |  |  |  |
| validation | Candidate |  |  |  |  |  |  |  |  |
| benchmark | Champion |  |  |  |  |  |  |  |  |
| benchmark | Candidate |  |  |  |  |  |  |  |  |
| hard_case_set | Champion |  |  |  |  |  |  |  |  |
| hard_case_set | Candidate |  |  |  |  |  |  |  |  |

同时输出 Candidate 相比 Champion 的变化：

| 数据集 | 指标 | Champion | Candidate | 差异 |
| --- | --- | --- | --- | --- |
| new_train | Recall |  |  |  |
| validation | mAP50-95 |  |  |  |
| benchmark | Recall |  |  |  |
| benchmark | FP |  |  |  |
| benchmark | FN |  |  |  |
| hard_case_set | Recall |  |  |  |

### 7.2 每类别指标

每个数据集都需要输出 per-class 指标：

| 类别 | GT 数量 | Precision | Recall | AP50 | AP50-95 | FP | FN | Candidate-Champion Recall 差异 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class_1 |  |  |  |  |  |  |  |  |
| class_2 |  |  |  |  |  |  |  |  |

当出现以下情况时，在报告中标记为风险项：

- 关键类别 Recall 下降超过配置阈值；
- 关键类别 FN 增加；
- 类别 AP50 或 AP50-95 明显下降；
- 类别 FP 显著增加；
- 类别样本量太少，评估不稳定；
- 高置信度 FP 增加；
- 小目标 FN 增加。

## 8. 人工审核与修标复核

### 8.1 人工审核方式

第一版不设计复杂的人工审核任务队列、工单状态或优先级流转机制。

每轮训练后，系统将各数据集中的 FP、FN、Candidate 新增错误和 Candidate 改进样本分别可视化输出。人工直接按数据集浏览这些图像即可。

人工检查的主要目的：

1. 判断 FP 是否是真实误检；
2. 判断 FN 是否是真实漏检；
3. 发现 FP 中是否存在标签漏标；
4. 判断 FN 是否来自标签错误、框偏移或目标难度；
5. 将确认的问题类型记录到审核结果；
6. 修正标签后生成新的 `dataset_version`。

### 8.2 review_result

第一版人工审核结果只强制记录问题类型，不强制记录处理动作。

建议输出为 CSV 或 JSON，核心字段：

```Plain Text
review_id
run_id
dataset_version
dataset_name
image_path
image_hash
error_type          # fp / fn / candidate_new_fp / candidate_new_fn / improved
model_role          # candidate / champion
class_name
confidence
problem_type        # label_error / model_error / uncertain / ignore
review_note
review_time
```

后续如果要自动构造训练集，可再升级为记录处理动作，例如 `fix_label`、`add_positive`、`add_negative`、`ignore`。

### 8.3 修标后复核报告

人工审核并修正标签后，必须生成新的 `dataset_version`，然后基于原始 `prediction_cache` 重新匹配 GT 并重算指标。

复核约束：

- 不覆盖原始报告；
- 不重新推理；
- 不改变 Candidate/Champion 模型；
- 不改变 eval_protocol；
- 只允许 GT 和 dataset_version 变化；
- 复核报告与原始报告保持独立，不强制做差异对比。

推荐命名：

```Plain Text
report_original.html
report_original.pdf
report_review_dataset_v2.html
report_review_dataset_v2.pdf
```

复核报告用于回答：“原始错误中有多少来自标签问题，修标后 Candidate/Champion 的指标分别是多少。”它不能直接证明模型能力提升，因为模型和预测结果没有变化。

## 9. 可视化与输出目录

### 9.1 输出目录结构

```Plain Text
runs/
└── model_v1.3.0/
    ├── manifests/
    │   ├── dataset_manifest_dataset_v1.json
    │   ├── dataset_manifest_dataset_v2.json
    │   └── run_manifest.json
    │
    ├── cache/
    │   ├── candidate_predictions.jsonl
    │   └── champion_predictions.jsonl
    │
    ├── metrics/
    │   ├── metrics_summary_original.csv
    │   ├── per_class_metrics_original.csv
    │   ├── candidate_vs_champion_original.csv
    │   ├── sample_level_results_original.csv
    │   ├── metrics_summary_review_dataset_v2.csv
    │   └── per_class_metrics_review_dataset_v2.csv
    │
    ├── review/
    │   └── review_result.csv
    │
    ├── visualizations/
    │   ├── <dataset_name>/
    │   │   ├── train/
    │   │   │   ├── fp/
    │   │   │   │   ├── images/
    │   │   │   │   ├── labels_gt/
    │   │   │   │   ├── labels_candidate/
    │   │   │   │   ├── labels_champion/
    │   │   │   │   └── compare/
    │   │   │   ├── fn/
    │   │   │   ├── candidate_new_fp/
    │   │   │   ├── candidate_new_fn/
    │   │   │   └── candidate_improved/
    │   │   └── val/
    │   │       ├── fp/
    │   │       ├── fn/
    │   │       ├── candidate_new_fp/
    │   │       ├── candidate_new_fn/
    │   │       └── candidate_improved/
    │   ├── <benchmark_dataset>/
    │   └── <hard_case_dataset>/
    │
    └── report/
        ├── report_original.html
        ├── report_original.pdf
        ├── report_review_dataset_v2.html
        └── report_review_dataset_v2.pdf
```

各目录说明：

| 文件夹 | 含义 |
| --- | --- |
| `fp/` | Candidate 模型的误检样本，包含可视化图、原图引用和标签引用 |
| `fn/` | Candidate 模型的漏检样本，包含可视化图、原图引用和标签引用 |
| `candidate_new_fp/` | Champion 没有误检，但 Candidate 新增误检 |
| `candidate_new_fn/` | Champion 检测正确，但 Candidate 漏检 |
| `candidate_improved/` | 按整张图判断，Candidate 的 `FP + FN` 少于 Champion 的样本 |

每个错误类型目录下统一保存：

```Plain Text
images/             # 原图
labels_gt/          # GT 标签
labels_candidate/   # Candidate 预测标签
labels_champion/    # Champion 预测标签；未配置 Champion 时为空或不生成
compare/            # 四栏对比图：原图 / GT / Champion / Candidate
```

说明：

1. FP 和 FN 分开保存，不再合并到同一目录，便于后续分别做误检治理和漏检治理；
2. 同一张图可能同时进入多个目录，例如既进入 `fp/`，也进入 `candidate_new_fp/`；
3. `save_diff: true` 表示生成上述 `visualizations/`，不再生成旧版 `diff/` 目录；
4. `save_diff: false` 表示只输出 `metrics/`、`cache/` 和 `manifests/`，不生成可视化样本。

### 9.2 FP/FN 可视化规范

每张 FP/FN 对比图应包含四栏：

```Plain Text
原图 | GT 标注 | Champion 预测 | Candidate 预测
```

颜色固定：

```Plain Text
GT：绿色
Champion：按匹配状态着色，不使用模型固定颜色
Candidate：按匹配状态着色，不使用模型固定颜色
FN 标记：橙色
FP 标记：橙色
TP 标记：绿色
预测文字：只显示置信度分数，不显示类别名或模型前缀
```

图像顶部可显示数据集、split 和错误类型：

```Plain Text
[candidate_new_fp]
dataset=benchmark
dataset_version=dataset_v1
champion_result=No FP
candidate_result=FP
```

人工主要判断：

1. 是否确实为模型误检；
2. 该位置是否存在漏标目标；
3. 是否是背景纹理、边界结构或相邻器件引起的误检；
4. Champion 是否存在同样问题；
5. 是否需要修正标签或补充样本。

### 9.3 Candidate 与 Champion 对比口径

当同时配置 Candidate 和 Champion 时，系统基于逐图结果生成新旧模型差异目录：

| 目录 | 判定条件 |
| --- | --- |
| `candidate_new_fp/` | Candidate 的 FP 数量大于 0，且 Champion 的 FP 数量等于 0 |
| `candidate_new_fn/` | Candidate 的 FN 数量大于 0，且 Champion 的 FN 数量等于 0 |
| `candidate_improved/` | `candidate_fp + candidate_fn < champion_fp + champion_fn` |

如果只配置 Candidate，不配置 Champion，则只生成 `fp/` 和 `fn/`。

## 10. 报告设计

### 10.1 原始验收报告

每轮自动验收输出原始报告：

```Plain Text
report_original.html
report_original.pdf
```

每个数据集章节开头先展示：

| 指标 | Champion | Candidate | 差异 |
| --- | --- | --- | --- |
| Precision |  |  |  |
| Recall |  |  |  |
| F1-score |  |  |  |
| mAP@0.5 |  |  |  |
| mAP@0.5:0.95 |  |  |  |
| FP 数量 |  |  |  |
| FN 数量 |  |  |  |

报告必须包含：

- `dataset_manifest` 摘要；
- `run_manifest` 摘要；
- eval_protocol 摘要；
- Candidate/Champion 总体对比；
- Candidate/Champion 每类别对比；
- 数据泄漏风险；
- FP/FN 样本入口；
- Candidate 新增错误样本入口；
- Candidate 改进样本入口；
- 自动上线建议。

### 10.2 复核报告

复核报告命名必须带新数据版本后缀：

```Plain Text
report_review_dataset_v2.html
report_review_dataset_v2.pdf
```

复核报告必须展示：

- 新 `dataset_version`；
- 使用的原始 `prediction_cache`；
- 标签修正后的 GT 数量变化；
- 重新匹配后的总体指标；
- 重新匹配后的每类别指标；
- 重新计算后的 FP/FN 统计。

复核报告不覆盖原始报告，也不替代下一轮训练报告。

## 11. 自动上线建议

系统可根据固定 Benchmark 和关键指标输出三种结论：

| 状态 | 含义 |
| --- | --- |
| `PASS` | 可进入灰度发布或正式上线评审 |
| `MANUAL_REVIEW` | 总体指标提升，但需人工重点检查 Benchmark 或 Validation 中的新增 FP/FN |
| `REJECT` | Benchmark 或关键类别明显退化，不建议上线 |

第一版只输出建议，不强制阻断发布。

默认阈值建议如下，项目可以在配置中覆盖：

```Plain Text
1. Benchmark mAP50-95 不低于 Champion；
2. Benchmark Recall 下降不超过 0.5 个百分点；
3. 关键类别 Recall 下降不超过 0.5 个百分点；
4. 关键类别 FN 不增加，或增加量不超过配置容忍值；
5. Benchmark FP 增长不超过 5%；
6. 高置信度 FP 不明显增加；
7. Hard Case Set 不出现明显退化；
8. 推理延迟 P95 不超过 Champion 的 110%；
9. benchmark/hard_case 数据泄漏风险必须在报告中显式标记；
10. 人工查看 Benchmark 和 Validation 的新增 FP/FN 后无明显问题。
```

默认建议逻辑：

```Plain Text
PASS:
    Benchmark 主要指标不低于 Champion，
    关键类别无明显退化，
    FP/FN 未超过阈值，
    无严重数据泄漏风险。

MANUAL_REVIEW:
    总体指标提升但存在新增 FP/FN，
    或存在轻微数据泄漏风险，
    或关键类别样本量太少导致评估不稳定。

REJECT:
    Benchmark mAP50-95 明显下降，
    或关键类别 Recall 明显下降，
    或 Benchmark FP/FN 超过阈值，
    或 hard_case_set 出现严重退化。
```

自动结论示例：

```Plain Text
总体结论：

Candidate 在固定 Benchmark 上优于 Champion。
Benchmark 的 mAP@0.5:0.95 提升 1.7 个百分点，
Recall 提升 1.7 个百分点，
FP 减少 19 个，FN 减少 28 个。

新增训练集上 Candidate Recall 明显高于 Champion，
说明新增数据已被较充分学习。
但新增训练集中仍存在若干 FN，建议人工检查其中是否包含小目标、
密集区域目标或标签问题。

当前自动建议：MANUAL_REVIEW。
建议人工重点查看：
1. benchmark/candidate_new_fn/
2. benchmark/candidate_new_fp/
3. validation/candidate_new_fn/
4. validation/candidate_new_fp/
```

## 12. 命令行接口建议

训练结束后，通过单一命令触发完整验收。每一次迭代都通过配置文件注入参数，执行后自动生成某次实验的所有结果文件夹。

示例：

```Shell
python scripts/run_acceptance.py \
  --candidate runs/train/exp/best.pt \
  --champion models/champion.pt \
  --data configs/yolo_data.yaml \
  --train-config configs/train.yaml \
  --eval-protocol configs/eval_protocol.yaml \
  --pretrained models/pretrained.pt \
  --output runs/model_v1.3.0
```

人工修标后，只重算指标并生成复核报告：

```Shell
python scripts/run_review_recheck.py \
  --run runs/model_v1.3.0 \
  --review-result runs/model_v1.3.0/review/review_result.csv \
  --updated-data configs/yolo_data_review_dataset_v2.yaml \
  --dataset-version dataset_v2
```

`run_review_recheck.py` 必须复用原始 `prediction_cache`，不得重新推理。

## 13. 推荐代码模块划分

```Plain Text
project/
├── configs/
│   ├── yolo_data.yaml
│   ├── train.yaml
│   └── eval_protocol.yaml
│
├── scripts/
│   ├── run_acceptance.py
│   ├── build_dataset_manifest.py
│   ├── train_model.py
│   ├── run_inference.py
│   ├── evaluate_dataset.py
│   ├── match_predictions_gt.py
│   ├── compare_models.py
│   ├── generate_visualizations.py
│   ├── generate_report.py
│   ├── run_deployment_gate.py
│   └── run_review_recheck.py
│
└── runs/
    ├── model_v1.3.0
    └── model_v1.4.0
```

模块职责：

| 模块 | 职责 |
| --- | --- |
| `run_acceptance.py` | 串联训练后验收主流程 |
| `build_dataset_manifest.py` | 读取 YOLO config 和目录，生成 `dataset_manifest` |
| `train_model.py` | 执行训练并保存模型和训练信息 |
| `run_inference.py` | 统一运行 Candidate/Champion 推理并保存 `prediction_cache` |
| `evaluate_dataset.py` | 计算 Precision、Recall、mAP 等指标 |
| `match_predictions_gt.py` | 计算 TP、FP、FN 和匹配关系 |
| `compare_models.py` | 比较 Candidate 与 Champion 的逐样本结果 |
| `generate_visualizations.py` | 生成 FP、FN、新增错误和改进样本图 |
| `generate_report.py` | 输出 HTML、PDF、CSV、JSON 报告 |
| `run_deployment_gate.py` | 输出模型上线建议 |
| `run_review_recheck.py` | 基于新数据版本和原始预测缓存重算指标，生成复核报告 |

## 14. 第一版验收标准

第一版系统完成后，至少应满足：

1. 可以从 YOLO data config 自动生成 `dataset_manifest`；
2. 每次训练或验收都能生成 `run_manifest`；
3. Candidate 和 Champion 使用同一 eval_protocol 推理；
4. 原始预测结果被保存为 `prediction_cache`；
5. 每个数据集都能输出总体指标、每类别指标和样本级 FP/FN；
6. 报告能展示 Candidate 相对 Champion 的新增错误和改进样本；
7. 报告能显式标记 benchmark/hard_case 泄漏风险；
8. 人工审核结果能记录问题类型；
9. 修标后能生成新的 `dataset_version`；
10. 复核报告能复用原始预测缓存重新匹配并重算指标；
11. 原始报告和复核报告互不覆盖；
12. 自动上线建议能输出 `PASS`、`MANUAL_REVIEW` 或 `REJECT`。
