import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import os

# --- 設定 ---
# 1. Config 檔案 (我們用的那個)
config_file = 'kitti_config.py'
# 2. 訓練好的權重 (請確認這個路徑存在)
checkpoint_file = 'work_dirs/deformable_detr_kitti/epoch_4.pth'
# 3. 測試圖片 (我們從訓練集中隨便挑一張，或者您可以換成網路上找的自駕車圖片)
img_path = 'data/kitti/images/train/000008.png' 
# 4. 輸出檔案名稱
out_file = 'result_kitti.jpg'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- 初始化模型 ---
print(f"正在載入模型... (Device: {device})")
model = init_detector(config_file, checkpoint_file, device=device)

# --- 進行推論 ---
print(f"正在處理圖片: {img_path}")
result = inference_detector(model, img_path)

# --- 視覺化 ---
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# 讀取圖片
img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

# 繪製結果
visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    wait_time=0,
    pred_score_thr=0.3 # 只顯示信心度 > 0.3 的框 (太低的過濾掉)
)

# 儲存圖片
visualizer.get_image()
img_out = visualizer.get_image()
mmcv.imwrite(mmcv.imconvert(img_out, 'rgb', 'bgr'), out_file)

print(f"✅ 成功！結果已儲存為 {out_file}，請打開來看看！")