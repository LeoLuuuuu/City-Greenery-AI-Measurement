import os
import cv2
import numpy as np

# ==============================================================
# 🚀 核心修复：OpenCV 中文路径支持补丁 (Monkey Patch)
# 将这部分放在最前面，提前给 cv2 的读写功能做“手术”
# ==============================================================
_original_imread = cv2.imread
_original_imwrite = cv2.imwrite


def imread_chinese(filename, flags=cv2.IMREAD_COLOR):
    try:
        # 使用 numpy 将图片读取为字节流，再由 cv2 解码，绕过中文路径报错
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
        return img if img is not None else _original_imread(filename, flags)
    except:
        return _original_imread(filename, flags)


def imwrite_chinese(filename, img, params=None):
    try:
        # 同样，在保存生成的图片时也绕过中文路径限制
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            n.tofile(filename)
            return True
        return _original_imwrite(filename, img, params)
    except:
        return _original_imwrite(filename, img, params)


# 强制替换底层的读取/写入方法
cv2.imread = imread_chinese
cv2.imwrite = imwrite_chinese
# ==============================================================

from zensvi.cv import Segmenter

# 1. 设定根目录
base_input_dir = r"D:\TreeSegmentation\test"
base_output_img_dir = r"D:\TreeSegmentation\segmented_images"
base_output_summary_dir = r"D:\TreeSegmentation\seg_summary"

# 2. 初始化语义分割模型
segmenter = Segmenter(dataset="mapillary", task="semantic")

print(f"🔍 开始扫描目录: {base_input_dir}")

# 3. 遍历 test 文件夹下的每一个子文件夹
for folder_name in os.listdir(base_input_dir):
    input_dir = os.path.join(base_input_dir, folder_name)

    if os.path.isdir(input_dir):
        output_img_dir = os.path.join(base_output_img_dir, folder_name)
        output_summary_dir = os.path.join(base_output_summary_dir, folder_name)

        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_summary_dir, exist_ok=True)

        print(f"\n🚀 正在处理子文件夹: {folder_name} ...")

        try:
            segmenter.segment(
                dir_input=input_dir,
                dir_image_output=output_img_dir,
                dir_summary_output=output_summary_dir,
                save_format="csv"
            )
            print(f"✅ {folder_name} 处理完成！")
        except Exception as e:
            print(f"❌ 处理 {folder_name} 时发生错误: {e}")

print("\n🎉 所有子文件夹的语义分割任务全部完成！")