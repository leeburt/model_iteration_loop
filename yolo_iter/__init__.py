"""YOLO pose 模型训练后自动验收与自迭代工具包。

提供以下子模块：
- config: 统一配置加载（project.yaml → train/acceptance profile）
- logging_utils: 日志工具
- manifest: 数据集清单生成与版本管理
- paths: 数据集路径解析
- pose_io: YOLO pose 标签读写与格式转换
- pose_tiny_match: 关键点感知的 GT-Pred 匹配算法与评估
- training: YOLO pose 训练封装
- acceptance: 完整验收流程编排
- recheck: 基于缓存预测结果的复核评估
"""

__all__ = [
    "config",
    "logging_utils",
    "manifest",
    "pose_tiny_match",
    "training",
]
