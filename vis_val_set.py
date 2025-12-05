import os
import json
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from tqdm import tqdm

# ================= 設定區 (請修改這裡) =================
# for kitti val set visualization
# 1. Config 檔案
config_file = 'kitti_config.py'

# 2. 權重檔
checkpoint_file = 'work_dirs/deformable_detr_kitti/epoch_50.pth'

# 3. 驗證集的 JSON 名單 (程式會從這裡知道該跑哪幾張圖)
val_json_file = 'data/kitti/annotations/kitti_val.json'

# 4. 圖片所在的真實資料夾 (KITTI 的話，驗證圖其實混在 train 資料夾裡)
img_root = 'D:/ROB 535 HW/detr_facebook/mmdetection/data/kitti/images/train'

# 5. 輸出結果存到哪
output_folder = 'vis_results_kitti'

# 6. 信心門檻
score_thr = 0.3
# =======================================================

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用裝置: {device}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 讀取 JSON 名單
    print(f"📖 正在讀取名單: {val_json_file}...")
    with open(val_json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 提取所有驗證集圖片的檔名
    # COCO JSON 結構: data['images'] 是一個 list，裡面有 {'file_name': 'xxx.png', ...}
    target_images = [img_info['file_name'] for img_info in coco_data['images']]
    
    print(f"✅ 名單讀取完畢，共有 {len(target_images)} 張驗證圖片。")

    # 2. 載入模型
    try:
        model = init_detector(config_file, checkpoint_file, device=device)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 3. 開始推論 (為了節省時間，我們只跑前 50 張，想跑全部請拿掉 [:50])
    print("🎨 開始繪製圖片...")
    for i, file_name in enumerate(tqdm(target_images[:50])):
        
        # 組合完整路徑
        img_path = os.path.join(img_root, file_name)
        
        # 檢查圖片是否存在
        if not os.path.exists(img_path):
            print(f"⚠️ 找不到圖片: {img_path}，跳過。")
            continue

        # 推論
        result = inference_detector(model, img_path)
        
        # 讀圖
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        # 繪圖
        visualizer.add_datasample(
            name=file_name,
            image=img,
            data_sample=result,
            draw_gt=False, # 如果設為 True，它會把標準答案(綠框)也畫上去，方便對比！
            show=False,
            pred_score_thr=score_thr
        )
        
        # 存檔
        out_file_path = os.path.join(output_folder, file_name)
        res_img = visualizer.get_image()
        res_img = mmcv.imconvert(res_img, 'rgb', 'bgr')
        mmcv.imwrite(res_img, out_file_path)

    print(f"✅ 全部完成！結果已儲存至: {output_folder}")

if __name__ == '__main__':
    main()