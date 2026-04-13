# 众智绘绿 (City-Greenery-AI-Measurement)

本项目是北京林业大学大学生创新创业训练计划（大创）研究成果。我们开发了一套基于深度学习（大视觉模型）的城市绿化自动化感知与测量管线。

## 🌟项目简介
本项目利用众源街景数据，结合前沿的AI感知技术，实现了从宏观绿视率统计到微观单木参数提取的全流程分析，旨在为城市森林资源管理和生态评估提供数字化支撑。

## 🛠️ 技术栈
- 语言: Python 3.8+
- 深度学习框架: PyTorch
- 核心模型: GroundingDINO, Segment Anything (SAM), Zensvi (Mapillary)
- 数据处理: OpenCV, Pandas, NumPy

## ✨ 核心功能
1. 街道绿视率 (GVI) 批量提取: 
   - 自动进行街景语义分割，识别植被要素。
   - 批量计算 GVI 指数并导出 CSV 报表。
2. 单木精细化实例分割:
   - 利用开放词汇检测 (GroundingDINO) 结合 SAM 抠出每一棵树。
   - 自研逻辑: 实现了树冠与树干的垂直合并算法，并自动提取单木宽度等几何参数。
3. 综合建模：SGQS 质量评估算法
   - 融合物理遮荫潜能 (Object-based) 与视觉感受 (Area-based) 两个维度。
   - 采用Z-score标准化消除量纲差异，支持多权重（0.7/0.3、0.6/0.4等）方案评价。

## 📂 项目结构
```text
├── core/
│   ├── instance_segmentation.py  # 单木实例分割主程序
│   ├── semantic_segmentation.py  # 绿视率语义分割脚本
│   ├── calculate_gvi.py          # GVI数值统计工具
│   └── calculate_sgqs.py         # SGQS 综合建模分析脚本
├── test/                         # 存放原始测试街景图
├── models/         #存放模型 
├── requirements.txt              # 环境依赖列表
└── README.md                     # 项目说明文档

## 📈 建模设计逻辑
我们提出的综合评分公式如下：$$SGQS = \alpha \cdot Zscore(\ln(S_{total})) + \beta \cdot Zscore(GVI_{area})$$
$\ln(S_{total})$: 代表物理遮荫潜能。取对数以平滑极端值，反映真实的生态覆盖。
$GVI_{area}$: 代表行人视觉体验，反映“看起来绿不绿”。
方案推荐: 采用 $\alpha=0.7, \beta=0.3$。该方案极大突出了深度学习矫正遥感数据后的物理价值，纠偏了传统评价中对物理遮荫能力的低估。


## 🚀 快速上手
### 1.配置环境:
```bash
pip install -r requirements.txt
```
2.准备权重:
由于权重文件较大，请从官方渠道下载 groundingdino_swint_ogc.pth 和 sam_vit_h_4b8939.pth 放入models/目录。
3.运行绿视率统计:
```bash
python core/semantic_segmentation.py
python core/calculate_gvi.py
```
4.运行单木参数提取:
```bash
python core/instance_segmentation.py
```
5.进行综合建模分析（SGQS）
```bash
python core/calculate_sgqs.py
```

##👥 团队与致谢
指导老师: 邹晟源老师
项目成员: 鲁东升，秦伟杰，冉从金，于佳玉，杨巧
依托单位: 北京林业大学信息学院

