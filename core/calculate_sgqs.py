import pandas as pd
import numpy as np
import os

# 1. 配置文件对，将参数表和距离表匹配，并给每个路段命名
file_pairs = [
    ("马连洼北路_1_tree_parameters.csv", "马连洼北路_1距离结果.csv", "马连洼北路_1"),
    ("马连洼北路_2_tree_parameters.csv", "马连洼北路_2距离计算结果.csv", "马连洼北路_2"),
    ("天秀南二路_tree_parameters.csv", "天秀南二路距离结果.csv", "天秀南二路"),
    ("天秀南一路_tree_parameters.csv", "天秀南一路距离结果.csv", "天秀南一路"),
    ("圆明园西路（北段）_tree_parameters.csv", "圆明园西路距离结果.csv", "圆明园西路")
]

summary_list_90 = []

# 2. 循环处理各个路段的数据文件
for param_file, dist_file, street_name in file_pairs:
    df_param = pd.read_csv(param_file)
    df_dist = pd.read_csv(dist_file)

    # 清理图片后缀，使得两个表格能够通过图片名称精准匹配
    df_param['原图名称'] = df_param['image_path'].str.replace('.jpg', '', regex=False).str.replace('.png', '',
                                                                                                   regex=False)
    df_param.rename(columns={'tree_id': '树编号'}, inplace=True)

    # 按照 原图名称 和 树编号 进行内连接合并
    df_merged = pd.merge(df_param, df_dist, on=['原图名称', '树编号'], how='inner')

    # 计算图像上的像素宽度 Delta X
    df_merged['Delta_x'] = df_merged['x_max'] - df_merged['x_min']

    # 核心步骤：使用 90度FOV 公式计算物理树宽 w = 2D * (Delta_x / W)
    df_merged['树冠宽度_w(90度FOV)'] = df_merged['估算距离(米)'] * (2 * 1.0 / df_merged['W']) * df_merged['Delta_x']

    # 假设俯视树冠为圆形，计算单株面积
    df_merged['树冠面积_90度(平方米)'] = np.pi * (df_merged['树冠宽度_w(90度FOV)'] / 2) ** 2

    # 按图片名称分组，求出单张图片内的树冠总面积
    grouped = df_merged.groupby('原图名称')['树冠面积_90度(平方米)'].sum().reset_index()
    grouped.rename(columns={'树冠面积_90度(平方米)': '单张图总树冠面积_90度'}, inplace=True)

    # 求自然对数，得到第一项得分 (加上极小值1e-9防止对0取对数报错)
    grouped['第一项得分_ln(总面积_90度)'] = np.log(grouped['单张图总树冠面积_90度'] + 1e-9)
    grouped.insert(0, '所属路段', street_name)

    summary_list_90.append(grouped)

# 3. 将 5 条路段的第一项得分（遮荫项）拼接到一个总表中
df_first_90 = pd.concat(summary_list_90, ignore_index=True)

# 4. 加载并在总表中合并第二项得分（绿视率 GVI）
if os.path.exists("all_gvi_scores.csv"):
    df_gvi = pd.read_csv("all_gvi_scores.csv")

    # 按照图片名称将绿视率合入总表
    df_final = pd.merge(df_first_90, df_gvi, left_on='原图名称', right_on='image_id', how='inner')

    # 5. 执行 Z-score 标准化操作（极重要：消除面积的ln值和绿视率百分比的量纲差异）
    df_final['第一项_Zscore(90度)'] = (df_final['第一项得分_ln(总面积_90度)'] - df_final[
        '第一项得分_ln(总面积_90度)'].mean()) / df_final['第一项得分_ln(总面积_90度)'].std()
    df_final['第二项_Zscore'] = (df_final['gvi_percent'] - df_final['gvi_percent'].mean()) / df_final[
        'gvi_percent'].std()

    # 6. 计算三种权重约束方案得分
    df_final['SGQS_方案1(0.7_0.3)_90度'] = 0.7 * df_final['第一项_Zscore(90度)'] + 0.3 * df_final['第二项_Zscore']
    df_final['SGQS_方案2(0.6_0.4)_90度'] = 0.6 * df_final['第一项_Zscore(90度)'] + 0.4 * df_final['第二项_Zscore']
    df_final['SGQS_方案3(0.5_0.5)_90度'] = 0.5 * df_final['第一项_Zscore(90度)'] + 0.5 * df_final['第二项_Zscore']

    # 7. 保存单图级别的结果明细表
    df_final.to_csv("所有路段_单图_三种方案SGQS得分_90度FOV.csv", index=False, encoding='utf-8-sig')

    # 8. 分组聚合生成街道级的最终排名汇总表
    street_summary_90 = df_final.groupby('所属路段').agg(
        平均_第一项_ln值_90度=('第一项得分_ln(总面积_90度)', 'mean'),
        平均_第二项_绿视率=('gvi_percent', 'mean'),
        SGQS_方案1_平均_90度=('SGQS_方案1(0.7_0.3)_90度', 'mean'),
        SGQS_方案2_平均_90度=('SGQS_方案2(0.6_0.4)_90度', 'mean'),
        SGQS_方案3_平均_90度=('SGQS_方案3(0.5_0.5)_90度', 'mean')
    ).reset_index()

    # 按方案1(核心推荐)的得分从高到低排列名次
    street_summary_90 = street_summary_90.sort_values(by='SGQS_方案1_平均_90度', ascending=False)

    # 保存街道整体汇总表
    street_summary_90.to_csv("所有路段_整体_三种方案SGQS汇总_90度FOV.csv", index=False, encoding='utf-8-sig')
    print("全部文件计算并生成成功！")
else:
    print("错误: 工作区找不到 all_gvi_scores.csv 绿视率文件。")