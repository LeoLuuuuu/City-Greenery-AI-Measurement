import os
import sys
import warnings
import cv2
import numpy as np
import torch
import csv
from tqdm import tqdm

# 1. 环境配置
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 优化显存分配策略，减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
grounding_dino_path = os.path.join(base_dir, "Grounded-Segment-Anything-main", "GroundingDINO")
segment_anything_path = os.path.join(base_dir, "Grounded-Segment-Anything-main", "segment_anything")

if grounding_dino_path not in sys.path:
    sys.path.insert(0, grounding_dino_path)
if segment_anything_path not in sys.path:
    sys.path.insert(0, segment_anything_path)

from groundingdino.util import box_ops
from torchvision.ops import nms
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict
from segment_anything import build_sam, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)


# -----------------------------
# 2. 模型加载
# -----------------------------
def load_model_local(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    return model


print("Loading models...", flush=True)
config_path = os.path.join(grounding_dino_path, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
checkpoint_path = os.path.join(base_dir, "groundingdino_swint_ogc.pth")
groundingdino_model = load_model_local(config_path, checkpoint_path, device)

sam_checkpoint = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)  # 回归 Predictor 模式，省显存
print("Models loaded successfully.", flush=True)


# -----------------------------
# 3. 核心工具函数
# -----------------------------

def consolidate_vertical_boxes(boxes, w_img, h_img, x_threshold=0.03, y_threshold=0.30):
    """
    【修复ID分裂】垂直合并算法。
    如果两棵“树”在X轴上对得很齐(偏差<10%)，就认为它们是同一棵树的上下部分，强制合并。
    """
    if len(boxes) == 0:
        return boxes

    # 转为列表处理
    box_list = [box.tolist() for box in boxes]
    # 按 X 轴中心排序
    box_list.sort(key=lambda x: x[0])

    merged_boxes = []
    while len(box_list) > 0:
        current = box_list.pop(0)
        cx1, cy1, w1, h1 = current

        # 计算当前框的边界
        x1_min, x1_max = cx1 - w1 / 2, cx1 + w1 / 2
        y1_min, y1_max = cy1 - h1 / 2, cy1 + h1 / 2

        # 寻找可合并的后续框
        i = 0
        while i < len(box_list):
            cx2, cy2, w2, h2 = box_list[i]

            # 检查水平对齐程度
            dist_x = abs(cx1 - cx2)
            dist_y = abs(cy1 - cy2)  # 新增 Y 轴距离判定

            if dist_x < x_threshold and dist_y < y_threshold:
                # 满足合并条件：取并集
                x2_min, x2_max = cx2 - w2 / 2, cx2 + w2 / 2
                y2_min, y2_max = cy2 - h2 / 2, cy2 + h2 / 2

                x1_min = min(x1_min, x2_min)
                x1_max = max(x1_max, x2_max)
                y1_min = min(y1_min, y2_min)
                y1_max = max(y1_max, y2_max)

                # 移除已被合并的框
                box_list.pop(i)
            else:
                i += 1

        # 反算合并后的中心点和宽高
        new_w = x1_max - x1_min
        new_h = y1_max - y1_min
        new_cx = x1_min + new_w / 2
        new_cy = y1_min + new_h / 2

        merged_boxes.append([new_cx, new_cy, new_w, new_h])

    return torch.tensor(merged_boxes, device=boxes.device)


def get_smart_search_region(box, image_h, image_w):
    """
    【智能搜索区】基于树干框，生成一个足够大的区域给SAM去发挥。
    """
    cx, cy, w, h = box.unbind(-1)
    new_w = w * 2.8  # 宽度放宽
    new_h = h * 2.5  # 高度放宽
    original_bottom = cy + h / 2
    new_cy = original_bottom - (new_h * 0.45)  # 保持底部基本不动，向上生长
    return torch.stack([cx, new_cy, new_w, new_h], dim=-1)


def analyze_mask_for_parameters(mask):
    """分析 Mask 提取形态参数"""
    if not np.any(mask): return None

    # 闭运算：连接断裂的树枝
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1: return None

    # 取面积最大的部分
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = (labels == max_label)

    # 计算边界
    h, w = final_mask.shape
    columns = np.where(final_mask)[1]
    if columns.size == 0: return None

    x_min, x_max = np.min(columns), np.max(columns)

    # 计算几何参数
    return {
        "W": w,
        "Y": w - x_max,
        "Y_prime": w - x_min,
        "x_min": x_min,
        "x_max": x_max
    }, final_mask

def cv2_imwrite_chinese(filepath, img):
    """
    【修复中文乱码】使用 imencode 和原生文件操作绕过 cv2.imwrite 的中文路径 bug
    """
    # 获取文件后缀名（如 .png 或 .jpg）
    ext = os.path.splitext(filepath)[1]
    # 将图像编码为对应格式的字节流
    is_success, im_buf_arr = cv2.imencode(ext, img)
    if is_success:
        im_buf_arr.tofile(filepath)
    else:
        print(f"Failed to encode image: {filepath}")

# -----------------------------
# 4. 主处理流程
# -----------------------------
def process_images(input_dir, output_dir):
    # --- 创建分类输出文件夹 ---
    detected_dir = os.path.join(output_dir, "detected_trees")  # 存放检测结果图
    debug_dir = os.path.join(output_dir, "debug_trees")  # 存放单独抠图
    mask_dir = os.path.join(output_dir, "tree_masks")  # 存放黑白掩膜

    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    # -----------------------

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    csv_path = os.path.join(output_dir, "tree_parameters.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=['image_path', 'tree_id', 'W', 'Y', 'Y_prime', 'x_min', 'x_max'])
        writer.writeheader()

        for img_path in tqdm(image_paths, desc="Processing Images"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # 手动释放显存
            torch.cuda.empty_cache()

            image_source, image = load_image(img_path)
            h_img, w_img, _ = image_source.shape

            # Step 1: GroundingDINO 检测树干
            # prompt = "tree trunk . fence . railing . pole"
            # prompt = "tree trunk . individual tree canopy"
            # prompt = "tree . tree trunk"
            prompt = "tree trunk . tree canopy"
            with torch.no_grad():  # 禁用梯度计算，省显存
                boxes, logits, phrases = predict(
                    model=groundingdino_model,
                    image=image,
                    caption=prompt,
                    box_threshold=0.18,
                    text_threshold=0.18,
                    device=device
                )

            if len(boxes) == 0: continue

            # Step 2: 终极几何特征审查 (专杀楼房和天空)
            filtered_indices = []
            for i, (box, phrase) in enumerate(zip(boxes, phrases)):
                # 提取宽、高、面积（归一化数值 0~1）
                w, h = box[2].item(), box[3].item()
                area = w * h

                # 规则 1：绝杀横向发展的干扰物（如汽车、长条围栏）
                if w > h * 1.2:
                    continue

                # 规则 2：体型分流审查 —— 你是树干，还是树冠？
                if h > w * 1.8:
                    # 【树干特征】：高度至少是宽度的 1.8 倍
                    # 真正的树干可以很高，但绝不可能很宽。
                    # 如果一个瘦长框的宽度竟然超过了画面的 30%（比如 w=0.4, h=0.8），那绝对是大楼的侧面截面！
                    if w > 0.30:
                        continue
                else:
                    # 【树冠特征】：长宽比例较接近（不满足 h > w * 1.8）
                    # 既然不是细长的树干，那就是飘在空中的无主干树冠。
                    # 街景中单棵树冠往往偏圆，且面积紧凑。
                    # 如果它面积巨大（超过20%）或者非常宽（超过45%），那绝对是把整栋楼或整片天空框进去了！
                    if area > 0.20 or w > 0.45:
                        continue

                filtered_indices.append(i)

            if not filtered_indices: continue

            indices = torch.tensor(filtered_indices, dtype=torch.long)
            boxes = boxes[indices]
            logits = logits[indices]

            # Step 3: NMS 去重
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
            keep = nms(boxes_xyxy, logits, iou_threshold=0.65)
            boxes = boxes[keep]

            # Step 4: 垂直合并
            boxes = consolidate_vertical_boxes(boxes, w_img, h_img, x_threshold=0.10)

            # Step 5: SAM 分割 (显存安全模式)
            sam_predictor.set_image(image_source)
            summary_overlay = image_source.copy()

            # 生成宽松的搜索区域
            search_boxes = get_smart_search_region(boxes, h_img, w_img)

            for i, (trunk_box, search_box) in enumerate(zip(boxes, search_boxes)):
                try:
                    # A. 准备 Prompt：同时给 框 和 点
                    # 框 (大范围搜索区)
                    box_xyxy = box_ops.box_cxcywh_to_xyxy(search_box)
                    box_sam = (box_xyxy * torch.Tensor([w_img, h_img, w_img, h_img])).cpu().numpy()
                    x0, y0, x1, y1 = box_sam.astype(int)
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w_img, x1), min(h_img, y1)

                    # 点 (树干中心)
                    # trunk_cx, trunk_cy, _, trunk_h = trunk_box.unbind(-1)
                    # pt_x = int(trunk_cx * w_img)
                    # pt_y = int((trunk_cy + trunk_h / 2) * h_img) - 10
                    #
                    # # B. 预测 (Multimask)
                    # masks, scores, _ = sam_predictor.predict(
                    #     point_coords=np.array([[pt_x, pt_y]]),
                    #     point_labels=np.array([1]),  # 1=前景
                    #     box=np.array([x0, y0, x1, y1]),  # 限制搜索范围
                    #     multimask_output=True  # 输出3个层级的掩码
                    # )

                    # C. 智能选择：选最大的那个 (Whole Tree)
                    # # 过滤掉全图背景 (面积 > 80% 则认为是背景错误)
                    # valid_indices = [idx for idx, m in enumerate(masks) if np.sum(m) < (w_img * h_img * 0.80)]
                    # if not valid_indices: continue
                    #
                    # # 在合理的掩码中，找面积最大的 (往往代表整棵树，而不是局部树干)
                    # best_idx = valid_indices[np.argmax([np.sum(masks[idx]) for idx in valid_indices])]
                    # raw_mask = masks[best_idx]

                    # --- 核心修改区开始 ---
                    trunk_cx, trunk_cy, trunk_w, trunk_h = trunk_box.unbind(-1)
                    pt_x = int(trunk_cx * w_img)
                    pt_y_center = int(trunk_cy * h_img)

                    # 动态判断：这是树干框还是树冠框？
                    # 如果高度是宽度的 1.5 倍以上，判定为树干；否则判定为树冠
                    is_trunk = (trunk_h > trunk_w * 1.5)

                    if is_trunk:
                        # 树干模式：打两个正向点（树干 + 树冠）
                        pt_y_canopy = int((trunk_cy - trunk_h * 0.8) * h_img)
                        pt_y_canopy = max(0, pt_y_canopy)
                        input_points = [[pt_x, pt_y_center], [pt_x, pt_y_canopy]]
                        input_labels = [1, 1]
                    else:
                        # 树冠模式（无树干）：直接在树冠中心打一个正向点
                        input_points = [[pt_x, pt_y_center]]
                        input_labels = [1]

                    # 负向点逻辑保持不变（防楼房）
                    bg_x_left = int(x0 + (x1 - x0) * 0.15)
                    bg_x_right = int(x1 - (x1 - x0) * 0.15)
                    bg_y_top = int(y0 + (y1 - y0) * 0.15)

                    # 将负向点加入列表
                    input_points.extend([[bg_x_left, bg_y_top], [bg_x_right, bg_y_top]])
                    input_labels.extend([0, 0])

                    input_points = np.array(input_points)
                    input_labels = np.array(input_labels)
                    # --- 下方的 SAM 预测代码保持不变 ---

                    # B. 预测 (Multimask)
                    masks, scores, _ = sam_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        box=np.array([x0, y0, x1, y1]),
                        multimask_output=True
                    )

                    # C. 智能选择：恢复选面积最大的 (设置 40% 防爆上限)
                    # 1. 过滤掉面积超过全图 40% 的超大遮罩（街景图里，单棵树通常不会超过40%，只有楼房和天空才会）
                    valid_indices = [idx for idx, m in enumerate(masks) if np.sum(m) < (w_img * h_img * 0.50)]
                    if not valid_indices: continue

                    # 2. 核心改变：不再选“面积最大”，而是选“SAM 置信度得分最高”的遮罩
                    valid_scores = [np.sum(masks[idx]) for idx in valid_indices]
                    best_idx = valid_indices[np.argmax(valid_scores)]
                    raw_mask = masks[best_idx]

                    # D. 后处理
                    result = analyze_mask_for_parameters(raw_mask)
                    if not result: continue
                    params, final_mask = result
                    params.update({'image_path': os.path.basename(img_path), 'tree_id': i + 1})
                    writer.writerow(params)

                    # E. 绘图与保存
                    color = np.random.randint(50, 255, (3,)).tolist()

                    # 1. 汇总图
                    summary_overlay[final_mask] = summary_overlay[final_mask] * 0.4 + np.array(color,
                                                                                               dtype=np.uint8) * 0.6
                    rows = np.any(final_mask, axis=1)
                    cols = np.any(final_mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        cv2.rectangle(summary_overlay, (cmin, rmin), (cmax, rmax), color, 3)
                        cv2.putText(summary_overlay, f"ID: {i + 1}", (cmin, max(rmin - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                    # 2. Debug Tree (黄紫色)
                    debug_img = np.zeros_like(image_source)
                    debug_img[:] = (84, 1, 68)
                    debug_img[final_mask] = (0, 235, 255)
                    debug_filepath = os.path.join(debug_dir, f"{base_name}_ID{i + 1}_debug.png")
                    cv2_imwrite_chinese(debug_filepath, debug_img)

                    # 3. Tree Mask (高亮)
                    dark_bg = (image_source * 0.4).astype(np.uint8)
                    highlight_layer = image_source.copy()
                    orange_tint = np.array([0, 140, 255], dtype=np.uint8)
                    highlight_layer[final_mask] = cv2.addWeighted(image_source[final_mask], 0.6,
                                                                  np.full_like(image_source[final_mask], orange_tint),
                                                                  0.4, 0)
                    tree_mask_img = dark_bg
                    tree_mask_img[final_mask] = highlight_layer[final_mask]
                    mask_filepath = os.path.join(mask_dir, f"{base_name}_ID{i + 1}_mask.png")
                    cv2_imwrite_chinese(mask_filepath, tree_mask_img)

                except Exception as e:
                    print(f"Error processing tree {i}: {e}")
                    continue

            # 保存汇总图
            detected_filepath = os.path.join(detected_dir, f"{base_name}_detected.jpg")
            cv2_imwrite_chinese(detected_filepath, cv2.cvtColor(summary_overlay, cv2.COLOR_RGB2BGR))

            # 每张图处理完彻底清理显存
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # 1. 定义总的输入和输出根目录
    root_input_dir = os.path.join(base_dir, "test")
    root_output_dir = os.path.join(base_dir, "test_output_refined")

    # 如果总输出目录不存在，先创建它
    os.makedirs(root_output_dir, exist_ok=True)

    # 2. 遍历 test 文件夹下的所有子文件夹
    # os.listdir 会列出 test 下的 "马连洼北路_1", "天秀南二路" 等
    for folder_name in os.listdir(root_input_dir):
        sub_input_dir = os.path.join(root_input_dir, folder_name)

        # 3. 必须判断它是不是一个文件夹（防止误读可能存在的隐藏文件，如 .DS_Store）
        if os.path.isdir(sub_input_dir):
            print(f"\n" + "=" * 50)
            print(f"🚀 正在处理子文件夹: {folder_name}")
            print(f"=" * 50)

            # 4. 在输出根目录下，为这个子文件夹建一个同名的专属输出文件夹
            sub_output_dir = os.path.join(root_output_dir, folder_name)

            # 5. 调用你的主函数，将这个子文件夹的路径传进去
            process_images(sub_input_dir, sub_output_dir)

    print(
        f"\n🎉 全部 {len([d for d in os.listdir(root_input_dir) if os.path.isdir(os.path.join(root_input_dir, d))])} 个文件夹处理完毕！")
    print(f"所有分类结果已保存至: {root_output_dir}", flush=True)