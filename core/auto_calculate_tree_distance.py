import os
import re
import cv2
import numpy as np
import torch
import sys
import csv
from torchvision import transforms

# 1. 核心配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 需要处理的文件夹
IMAGE_FOLDER = r"D:\monodepth2\test5"
MODEL_DIR = r"D:\monodepth2\models\mono+stereo_1024x320"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pth")
DECODER_PATH = os.path.join(MODEL_DIR, "depth.pth")
MODEL_INPUT_SIZE = (1024, 320)
# 距离参数
CALIBRATION_FACTOR = 0.56

# 2. 导入模型
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.networks.depth_decoder import DepthDecoder


# 3. 加载模型
def load_monodepth2_model():
    encoder = ResnetEncoder(18, False)
    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=[0])
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE), strict=False)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE), strict=False)
    encoder.to(DEVICE).eval()
    decoder.to(DEVICE).eval()
    return encoder, decoder


# 4. 中文路径读取图片
def cv_imread(file_path):
    # 中文路径专用读取
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


# 5. 图片预处理
def preprocess_image(image_path):
    image = cv_imread(image_path)
    if image is None:
        return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, MODEL_INPUT_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_resized).unsqueeze(0).to(DEVICE)
    return image, image_tensor


# 6. 生成深度图
def generate_depth_map(image_tensor, original_h, original_w):
    with torch.no_grad():
        features = encoder(image_tensor)
        outputs = decoder(features)
        disparity_map = outputs["disp", 0].cpu().numpy().squeeze()
    depth_map = 1.0 / (disparity_map + 1e-6)
    depth_map_resized = cv2.resize(depth_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return depth_map_resized


# 7. 提取黄色掩码
def extract_yellow_tree_mask(mask_path, original_h, original_w):
    mask = cv_imread(mask_path)
    if mask is None:
        return None
    mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_mask, lower_yellow, upper_yellow)
    _, binary_mask = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


# 8. 计算距离
def calculate_tree_distance(depth_map, binary_mask):
    mask_coords = np.where(binary_mask == 255)
    if len(mask_coords[0]) == 0:
        return None, "无有效树木像素"
    depths = depth_map[mask_coords]
    median_depth = np.median(depths)
    mad = np.median(np.abs(depths - median_depth))
    if mad > 0:
        valid_mask = np.abs(depths - median_depth) <= 3 * mad
        valid_depths = depths[valid_mask]
    else:
        valid_depths = depths
    final_median = np.median(valid_depths)
    calibrated_distance = final_median * CALIBRATION_FACTOR
    return round(calibrated_distance, 2), {
        "总像素数": len(depths),
        "有效像素数": len(valid_depths),
        "中位数深度": round(float(final_median), 4)
    }


# 9. 解析文件名
def parse_mask_filename(mask_filename):
    name_no_ext = os.path.splitext(mask_filename)[0]
    if "_ID" in name_no_ext:
        original_name = name_no_ext.split("_ID")[0]
        tree_part = name_no_ext.split("_ID")[1]
        tree_id = tree_part.split("_")[0]
        return original_name, tree_id
    return None, None


# 10. 主流程
def main():
    global encoder, decoder
    encoder, decoder = load_monodepth2_model()
    print("模型加载完成，开始处理掩码...")

    result_csv = os.path.join(IMAGE_FOLDER, "树木距离结果.csv")
    with open(result_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["原图名称", "树编号", "掩码路径", "总像素数", "有效像素数", "中位数深度", "估算距离(米)", "状态"])

    mask_files = sorted(
        [f for f in os.listdir(IMAGE_FOLDER)
         if "_ID" in f and "_debug" in f and f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: parse_mask_filename(x)[0]
    )

    if not mask_files:
        print("未找到符合格式的掩码文件！格式：XXX_ID1_debug.png")
        return

    depth_cache = {}
    for mask_file in mask_files:
        mask_path = os.path.join(IMAGE_FOLDER, mask_file)
        original_name, tree_id = parse_mask_filename(mask_file)

        if not original_name or not tree_id:
            print(f"格式错误：{mask_file}")
            continue

        # 找原图
        original_image_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(IMAGE_FOLDER, f"{original_name}{ext}")
            if os.path.exists(candidate):
                original_image_path = candidate
                break
        if not original_image_path:
            print(f"未找到原图：{mask_file}")
            continue

        # 生成/使用缓存深度图
        if original_name not in depth_cache:
            original_image, image_tensor = preprocess_image(original_image_path)
            if original_image is None:
                print(f"原图读取失败：{original_image_path}")
                continue
            h, w = original_image.shape[:2]
            depth_map = generate_depth_map(image_tensor, h, w)
            depth_cache[original_name] = depth_map
        else:
            depth_map = depth_cache[original_name]
            h, w = depth_map.shape[:2]

        # 提取掩码
        binary_mask = extract_yellow_tree_mask(mask_path, h, w)
        if binary_mask is None:
            print(f"掩码读取失败：{mask_file}")
            continue

        # 计算距离
        distance, stats = calculate_tree_distance(depth_map, binary_mask)
        if distance is None:
            print(f"计算失败 {tree_id}：{stats}")
        else:
            print(f"{original_name} 树{tree_id}：{distance} 米")
            with open(result_csv, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(
                    [original_name, tree_id, mask_path, stats["总像素数"], stats["有效像素数"], stats["中位数深度"], distance, "成功"])

    print(f"\n处理完成！结果：{result_csv}")


if __name__ == "__main__":
    main()