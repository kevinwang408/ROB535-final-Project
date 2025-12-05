import os
import glob
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from tqdm import tqdm

# ================= 設定區 (請根據你要跑的資料集修改這裡) =================

# --- 設定 1: 跑 KITTI (取消註解這區塊) ---
# config_file = 'kitti_config.py'
# checkpoint_file = 'work_dirs/deformable_detr_kitti/epoch_50.pth' # 確保用 epoch_50
# input_folder = 'data/kitti/images/testing/image_2' # KITTI 測試集路徑
# output_folder = 'vis_results_kitti'                # 結果存到這裡
# img_ext = '*.png'

# --- 設定 2: 跑 BDD100K (取消註解這區塊) ---
config_file = 'bdd_ninja_config.py'
checkpoint_file = 'work_dirs/deformable_detr_bdd_ninja/epoch_50.pth'
input_folder = 'data/bdd100k_ninja/val/img' # 或者是 val/img，看你想測哪個
output_folder = 'vis_results_bdd'
img_ext = '*.jpg'

# 信心門檻 (0.3 代表信心度 > 30% 才畫出來，避免畫面太亂)
score_thr = 0.3
# =====================================================================

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用裝置: {device}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 載入模型
    print(f"⏳ 正在載入模型: {checkpoint_file}...")
    try:
        model = init_detector(config_file, checkpoint_file, device=device)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 2. 搜尋圖片
    search_path = os.path.join(input_folder, img_ext)
    images = glob.glob(search_path)
    print(f"📸 找到 {len(images)} 張圖片，準備開始推論...")

    # 3. 批量推論 (只跑前 50 張做展示即可，跑全部會很久)
    # 如果想跑全部，請把 [:50] 拿掉
    for i, img_path in enumerate(tqdm(images[:200])): 
        
        # 推論
        result = inference_detector(model, img_path)
        
        # 讀圖並轉檔
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb') 

        # 繪圖
        visualizer.add_datasample(
            name=os.path.basename(img_path),
            image=img,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thr
        )
        
        # 存檔
        out_file_path = os.path.join(output_folder, os.path.basename(img_path))
        res_img = visualizer.get_image()
        res_img = mmcv.imconvert(res_img, 'rgb', 'bgr')
        mmcv.imwrite(res_img, out_file_path)

    print(f"✅ 完成！結果已儲存在資料夾: {output_folder}")

if __name__ == '__main__':
    main()