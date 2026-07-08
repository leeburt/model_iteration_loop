"""YOLO 模型训练后自动验收与自迭代工具包。

提供以下子模块：
- config: 统一配置加载（project.yaml → train/acceptance profile）
- logging_utils: 日志工具
- manifest: 数据集清单生成与版本管理
- paths: 数据集路径解析
- detect_io: YOLO detect 标签读写与格式转换
- detect_match: bbox IoU 感知的 GT-Pred 匹配算法与评估
- evaluation: pose/detect 评估后端选择
- pose_io: YOLO pose 标签读写与格式转换
- pose_tiny_match: 关键点感知的 GT-Pred 匹配算法与评估
- training: YOLO 训练封装
- acceptance: 完整验收流程编排
- recheck: 基于缓存预测结果的复核评估
"""

__all__ = [
    "config",
    "detect_io",
    "detect_match",
    "evaluation",
    "logging_utils",
    "manifest",
    "pose_tiny_match",
    "training",
]
