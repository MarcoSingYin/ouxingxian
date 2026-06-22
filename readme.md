# 项目说明文档

## 一、项目概要

本项目为新人入职培训训练项目，旨在快速熟悉医疗数据处理、自然语言处理（实体抽取）以及机器学习建模的基本流程。项目整体分为三个部分：

- **TASK1**：基于大语言模型的病例实体提取程序，从PDF病例文本中自动抽取关键医学实体（如症状、诊断、用药等），输出结构化JSON结果。
- **TASK2**：基于机器学习的心衰患者死亡率预测项目，包含完整的数据分析、特征工程、多模型训练与评估，并生成一份符合学术规范的论文（LaTeX源码）。
- **TASK3**：医疗大模型新人快速上手指南，以Markdown和PDF形式提供，涵盖大模型基础、API调用、常见问题等，帮助新人快速上手。

### 仓库文件树及说明
```
ouxingxian/
├── TASK1/
│   ├── A case of portal vein recanalization and symptomatic heart failure.pdf   # 示例病例PDF（英文）
│   ├── requirements.txt                                                         # TASK1依赖库列表
│   ├── task1_2.0.py                                                             # TASK1主程序（实体抽取）
│   └── 提取病例实体2.0.json                                                     # 程序输出示例（结构化实体）
├── TASK2/
│   ├── Comparison of Machine Learning Models for All Cause Mortality Prediction in Heart Failure A Retrospective Cohort Analysis of Public Clinical Data.pdf   # 生成的论文PDF
│   ├── calibration_curve.png                                                    # 校准曲线图
│   ├── concept_diagram.png.png                                                  # 概念图（文件名有重复后缀，保留原样）
│   ├── correlation_heatmap.png                                                  # 特征相关性热力图
│   ├── flowchart.png                                                            # 研究流程图
│   ├── heart_failure_clinical_records_dataset.csv                               # 原始数据集（299条记录）
│   ├── lr_coefficient_importance.png                                            # 逻辑回归系数重要性图
│   ├── main.tex                                                                 # 论文LaTeX源码
│   ├── requirements.txt                                                         # TASK2依赖库列表
│   ├── roc_curves.png                                                           # 多模型ROC曲线对比图
│   ├── task2_5.0.py                                                             # TASK2主程序（完整建模流程）
│   ├── ttest_results.csv                                                        # t检验结果表
│   └── xgb_feature_importance_with_meaning.png                                  # XGBoost特征重要性（标注医学含义）
├── TASK3/
|   ├── 医疗大模型新人快速上手指南.md                                             # Markdown版指南
|   └── 医疗大模型新人快速上手指南.pdf                                             # PDF版指南
├──AI_Chat_Records.md
└──readme.md
```



---

## 二、运行方法

### 环境准备

1. 确保已安装 **Python 3.8+** 环境。
2. 分别进入 **TASK1** 和 **TASK2** 文件夹，根据各自提供的 **requirements.txt** 安装依赖包。  
   建议使用虚拟环境（如 **venv** 或 **conda**）隔离安装。
   ```bash
   cd TASK1
   pip install -r requirements.txt
   cd ../TASK2
   pip install -r requirements.txt
   
   > **注意**：TASK1 需要调用大语言模型API，请提前准备好API密钥（如OpenAI、豆包等），并确保网络通畅。

### 运行 TASK1（病例实体提取）

- 主程序：`task1_2.0.py`
- 运行方式：
  ```bash
  python task1_2.0.py
  ```
- 程序会提示输入以下信息（根据窗口提示操作）：
  - 大模型API密钥（或选择预设配置）
  - 待处理的PDF文件路径（默认使用同目录下的示例PDF）
  - 输出JSON文件名（默认生成 **提取病例实体2.0.json**）
- 程序运行后，会在同目录下生成包含实体提取结果的JSON文件，示例输出已提供。

### 运行 TASK2（心衰死亡预测）

- 主程序：`task2_5.0.py`
- 运行方式（直接运行，无需额外输入）：
  ```bash
  python task2_5.0.py
  ```
- 程序将依次执行：
  1. 数据加载与描述性统计
  2. 缺失值/异常值检测与处理（采用医学常识阈值封顶）
  3. 特征相关性分析与可视化（保存热力图）
  4. 特征重要性评估（XGBoost + 逻辑回归系数）
  5. 使用分层交叉验证选择最佳模型（Logistic Regression、Random Forest、XGBoost、MLP）
  6. 在保留测试集上评估最终模型，输出分类指标、ROC曲线、校准曲线
  7. 生成t检验结果表、特征重要性条形图（带医学含义）
  8. 输出所有图表和结果文件（均在 **TASK2** 文件夹下）

- 程序运行结束后，所有输出文件（图片、CSV、PDF论文等）将保存在当前目录，可直接查看。

### 示例输出

每个程序对应的示例输出文件均已包含在仓库对应文件夹中，可供参考对比：
- TASK1 示例输出：**提取病例实体2.0.json**
- TASK2 示例输出：所有图片文件、**ttest_results.csv**、**main.tex** 及生成的论文PDF。

---

## 三、注意事项

1. **路径问题**：所有文件均使用相对路径，请将 **TASK1** 和 **TASK2** 文件夹置于同一根目录下（或分别独立运行，程序会自动寻找同目录下的数据文件）。
2. **网络要求**：TASK1 需要调用大模型API，请确保API密钥有效且网络可访问对应服务。
3. **硬件要求**：TASK2 中的 XGBoost 和 PyTorch MLP 训练对CPU/内存要求不高（299条数据），普通笔记本电脑即可流畅运行。
4. **可复现性**：所有随机种子已在代码中固定，保证每次运行结果一致。

---

## 四、快速开始（新人建议）

1. 首先阅读 **TASK3/医疗大模型新人快速上手指南.pdf**，了解大模型基本概念和API调用方法。
2. 按顺序运行 TASK1，理解如何通过大模型从非结构化文本中提取结构化信息。
3. 运行 TASK2，观察完整的机器学习建模流程，并阅读生成的论文，学习如何将技术结果转化为学术报告。

如有任何问题，请参考各任务文件夹内的注释或联系项目导师。
