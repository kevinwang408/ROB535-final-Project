import os
import json
import glob
import cv2
import shutil
import random
from sklearn.model_selection import train_test_split

# --- 設定路徑 ---
ROOT_DIR = 'data/kitti'
IMAGE_DIR = os.path.join(ROOT_DIR, 'images/train')
LABEL_DIR = os.path.join(ROOT_DIR, 'labels')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'annotations')

# --- 設定類別 (您可以根據需求增減，KITTI 主要是這三類) ---
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CAT_MAP = {name: i + 1 for i, name in enumerate(CLASSES)}

def parse_kitti_label(label_file):
    boxes = []
    labels = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            cls_name = parts[0]
            
            # 只處理我們關心的類別，忽略 DontCare, Misc 等
            if cls_name in CLASSES:
                # KITTI 格式: type truncated occluded alpha x1 y1 x2 y2 ...
                # 我們只需要 x1, y1, x2, y2 (index 4, 5, 6, 7)
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
                
                boxes.append([x1, y1, x2, y2])
                labels.append(CAT_MAP[cls_name])
    return boxes, labels

def convert_to_coco(image_list, mode='train'):
    coco_dataset = {
        "info": {"description": "KITTI 2D converted to COCO"},
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in CAT_MAP.items()]
    }
    
    ann_id = 0
    print(f"正在轉換 {mode} 數據集，共 {len(image_list)} 張圖片...")

    for img_id, img_path in enumerate(image_list):
        # 讀取圖片資訊
        filename = os.path.basename(img_path)
        img_id_str = os.path.splitext(filename)[0]
        
        # 讀取圖片尺寸 (為了寫入 COCO info)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 無法讀取圖片 {img_path}")
            continue
        height, width, _ = img.shape
        
        coco_dataset["images"].append({
            "file_name": filename, # 相對於 images/train 的路徑
            "height": height,
            "width": width,
            "id": int(img_id_str) # 使用原始檔案名稱當 ID (例如 000001)
        })
        
        # 讀取對應的標籤
        label_file = os.path.join(LABEL_DIR, img_id_str + '.txt')
        if os.path.exists(label_file):
            boxes, class_ids = parse_kitti_label(label_file)
            
            for box, cls_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                coco_dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": int(img_id_str),
                    "category_id": cls_id,
                    "bbox": [x1, y1, w, h], # COCO 格式是 [x, y, w, h]
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
                
    # 儲存 json
    save_path = os.path.join(OUTPUT_DIR, f'kitti_{mode}.json')
    with open(save_path, 'w') as f:
        json.dump(coco_dataset, f)
    print(f"已儲存: {save_path}")

def main():
    # 1. 獲取所有圖片列表
    all_images = glob.glob(os.path.join(IMAGE_DIR, '*.png'))
    if not all_images:
        print("❌ 錯誤: 在 images/train 找不到任何 .png 圖片！請確認路徑。")
        return

    # 2. 切分訓練集 (80%) 和 驗證集 (20%)
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # 3. 執行轉換
    convert_to_coco(train_imgs, 'train')
    convert_to_coco(val_imgs, 'val')
    print("✅ 轉換完成！")

if __name__ == '__main__':
    main()