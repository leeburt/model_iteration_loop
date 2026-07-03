# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


# 项目目的

* 构建一个模型优化自迭代的系统，放入新增数据，给出最新模型的评估，人工check 训练和验证数据后再训练，这样就可以通过数据驱动完成模型的自动迭代

## 代码风格

- Python 3.x，使用类型注解（type hints）
- 配置通过 YAML 文件注入，禁止在脚本中硬编码路径和超参
- 所有路径操作使用 `pathlib.Path`
- 脚本间通过命令行参数传递配置，不通过全局变量或环境变量隐式通信
- 指标计算统一用 float，保留 4 位小数输出
- 可视化颜色固定：GT 绿色、Champion 蓝色、Candidate 红色、FP 红色框、FN 黄色框
- 如无必要勿增实体
- 每一个模块和函数都需要写注释，需要说明模块的作用，以及函数的作用函数的输入和输出
- 每个模块尽可能的解耦开，方便后续的迭代升级
- 不需要做复杂的封装
