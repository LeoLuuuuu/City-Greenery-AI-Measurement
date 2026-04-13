import os
import cv2
import numpy as np
import pandas as pd

#设定路径
base_segmented_dir = os.path.join(".", "segmented_images")
output_csv_path = os.path.join(".", "all_gvi_scores.csv")

#Mapillary vegetation 的 RGB 配色
VEGETATION_RGB = (107, 142, 35)

records = []

if not os.path.exists(base_segmented_dir):
    print(f"错误：找不到路径 {base_segmented_dir}")
else:
    print(f"正在从 {base_segmented_dir} 遍历所有子文件夹读取图片...")

    for folder_name in os.listdir(base_segmented_dir):
        folder_path = os.path.join(base_segmented_dir, folder_name)

        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if not fname.endswith("_colored_segmented.png"):
                    continue

                fpath = os.path.join(folder_path, fname)

                # 🚀 核心修改：使用 numpy 绕过 cv2 的中文路径读取限制
                img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    print(f"警告：无法读取图片 {fpath}")
                    continue

                # 转为 RGB 计算像素
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = np.all(img_rgb == VEGETATION_RGB, axis=2)
                vegetation_pixels = int(np.sum(mask))
                total_pixels = img.shape[0] * img.shape[1]

                if total_pixels == 0:
                    gvi_percent = 0.0
                else:
                    gvi_percent = round(vegetation_pixels / total_pixels * 100, 2)

                records.append({
                    "folder": folder_name,
                    "image_id": fname.replace("_colored_segmented.png", ""),
                    "vegetation_pixels": vegetation_pixels,
                    "total_pixels": total_pixels,
                    "gvi_percent": gvi_percent
                })

# 输出汇总表
if records:
    df = pd.DataFrame(records)
    # 使用 utf-8-sig 防止 Excel 乱码
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 提取完成，所有路段的结果已汇总保存至：{output_csv_path}")
    print("\n--- 预览前几行数据 ---")
    print(df.head())
else:
    print("未找到任何处理过的图片，请检查第三步是否成功运行。")