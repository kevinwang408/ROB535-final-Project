# 繼承官方的 Deformable DETR 設定 (ResNet-50 backbone)
_base_ = 'configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

# --- 1. 資料集路徑設定 ---
data_root = 'data/kitti/'
# 定義我們的類別 (必須與 convert_kitti.py 裡的一致)
class_name = ('Car', 'Pedestrian', 'Cyclist')
# 定義類別名稱和顯示顏色 (R, G, B)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142)])

# --- 2. 模型設定修改 ---
model = dict(
    bbox_head=dict(
        # 重要：修改類別數量為 3 (原本是 80)
        num_classes=3))

# --- 3. 資料載入器 (Data Loader) 修改 ---
train_dataloader = dict(
    batch_size=2,  # 如果您的 GPU 記憶體小於 8GB，請改成 1
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # 指向我們生成的 json
        ann_file='annotations/kitti_train.json',
        # 指向圖片資料夾
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/kitti_val.json',
        data_prefix=dict(img='images/train/')))

test_dataloader = val_dataloader

# --- 4. 評估標準設定 ---
val_evaluator = dict(
    ann_file=data_root + 'annotations/kitti_val.json',
    metric='bbox') # 使用 COCO 格式的 mAP 指標
test_evaluator = val_evaluator

# --- 5. 訓練時程設定 ---
# 為了快速看到結果，我們先設定跑 12 個 Epoch (原本是 50)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=1)

# 設定學習率 (如果 batch_size 改小了，學習率通常也要按比例縮小，這裡先用預設的一半)
optim_wrapper = dict(optimizer=dict(lr=0.0001))

# 設定儲存權重的路徑
work_dir = './work_dirs/deformable_detr_kitti'