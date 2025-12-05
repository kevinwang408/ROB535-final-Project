import cv2
import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import time
import os

# ================= 設定區 (請修改這裡) =================
# 1. Config 檔案名稱 (確保這跟您訓練用的一樣)
config_file = "D:/ROB 535 HW/detr_facebook/mmdetection/work_dirs/deformable_detr_kitti/kitti_config.py" 

# 2. 剛下載回來的權重檔路徑
checkpoint_file = "D:/ROB 535 HW/detr_facebook/mmdetection/work_dirs/deformable_detr_kitti/epoch_50.pth" # 請改成您的實際路徑

# 3. 要測試的影片檔 (請準備一個 mp4)
video_path = "D:/ROB 535 HW/detr_facebook/mmdetection/test_video/phili.mp4" 

# 4. 輸出結果檔名
out_path = "D:/ROB 535 HW/detr_facebook/mmdetection/video detection test/kitti/result_video.mp4"

# 5. 信心門檻 (0.3 代表 30% 把握才畫框，想看更多框可調低)
score_thr = 0.3
# =======================================================

def main():
    # 檢查檔案是否存在
    if not os.path.exists(checkpoint_file):
        print(f"❌ 錯誤：找不到權重檔 {checkpoint_file}")
        return
    if not os.path.exists(video_path):
        print(f"❌ 錯誤：找不到影片檔 {video_path}")
        return

    # 1. 初始化模型
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 正在載入模型至 {device} ...")
    
    try:
        model = init_detector(config_file, checkpoint_file, device=device)
    except Exception as e:
        print(f"❌ 模型載入失敗，請檢查 Config 與權重是否匹配。\n錯誤訊息: {e}")
        return

    # 2. 準備視覺化工具
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 3. 讀取影片
    video_reader = mmcv.VideoReader(video_path)
    width, height = video_reader.width, video_reader.height
    fps = video_reader.fps
    
    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"🎬 開始處理影片: {width}x{height}, FPS: {fps}, 總幀數: {len(video_reader)}")
    
    start_time = time.time()
    
    # 4. 逐幀推論
    for i, frame in enumerate(video_reader):
        # 推論
        result = inference_detector(model, frame)
        
        # 繪圖
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thr
        )
        frame_vis = visualizer.get_image()
        
        # 轉換顏色 (RGB -> BGR) 讓 OpenCV 正確存檔
        frame_vis = mmcv.imconvert(frame_vis, 'rgb', 'bgr')
        
        # 寫入
        video_writer.write(frame_vis)

        # 顯示進度條
        if (i + 1) % 10 == 0:
            print(f"⏳ 進度: {i + 1}/{len(video_reader)} 幀", end='\r')

    video_writer.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n✅ 完成！結果已儲存至: {out_path}")
    print(f"⏱️ 總耗時: {total_time:.2f} 秒 (平均 FPS: {len(video_reader)/total_time:.1f})")

if __name__ == '__main__':
    main()