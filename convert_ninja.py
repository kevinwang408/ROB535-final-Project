import os
import json
import glob
from tqdm import tqdm

# --- 設定路徑 ---
ROOT_DIR = 'data/bdd100k_ninja'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'annotations')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# BDD100K 的標準 10 類 (我們強制固定這個順序，以免 ID 亂掉)
CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", 
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]
# 建立名稱到 ID 的對照表 (1-based index)
CAT_MAP = {name: i + 1 for i, name in enumerate(CLASSES)}

def get_bbox_from_points(points):
    # 從多邊形點計算 [xmin, ymin, w, h]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def convert_folder(subset_name):
    print(f"🔄 正在轉換 {subset_name} 資料集...")
    
    ann_dir = os.path.join(ROOT_DIR, subset_name, 'ann')
    img_dir = os.path.join(ROOT_DIR, subset_name, 'img')
    
    # 找所有的 JSON 檔
    json_files = glob.glob(os.path.join(ann_dir, '*.json'))
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name} for name, id in CAT_MAP.items()]
    }
    
    ann_id_counter = 0
    img_id_counter = 0
    
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # 1. 處理圖片資訊
        # 檔名通常是 "abc.jpg.json"，我們要還原成 "abc.jpg"
        # 注意：要確認 img 資料夾裡的副檔名是 .jpg 還是 .png
        base_name = os.path.basename(json_file).replace('.json', '') 
        
        # 簡單檢查一下圖片是否存在 (防止副檔名對不上)
        if not os.path.exists(os.path.join(img_dir, base_name)):
            # 嘗試換副檔名找找看
            if os.path.exists(os.path.join(img_dir, base_name.replace('.jpg', '.png'))):
                base_name = base_name.replace('.jpg', '.png')
            elif os.path.exists(os.path.join(img_dir, base_name.replace('.png', '.jpg'))):
                 base_name = base_name.replace('.png', '.jpg')
        
        image_info = {
            "file_name": base_name,
            "height": data['size']['height'],
            "width": data['size']['width'],
            "id": img_id_counter
        }
        coco_output["images"].append(image_info)
        
        # 2. 處理標註資訊
        for obj in data['objects']:
            class_name = obj['classTitle']
            
            # 只處理我們定義的那 10 類
            if class_name in CAT_MAP:
                # 取得 bbox
                points = obj['points']['exterior']
                bbox = get_bbox_from_points(points) # [x, y, w, h]
                
                annotation = {
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": CAT_MAP[class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [sum(points, [])] # 簡單把點攤平
                }
                coco_output["annotations"].append(annotation)
                ann_id_counter += 1
                
        img_id_counter += 1
        
    # 儲存
    out_path = os.path.join(OUTPUT_DIR, f'bdd100k_{subset_name}.json')
    with open(out_path, 'w') as f:
        json.dump(coco_output, f)
    print(f"✅ 已儲存至: {out_path}")

if __name__ == '__main__':
    convert_folder('train')
    convert_folder('val')