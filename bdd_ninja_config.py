_base_ = 'configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

# --- 1. 定義 BDD100K 的 10 個類別與顏色 ---
classes = ('pedestrian', 'rider', 'car', 'truck', 'bus', 
           'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign')

# 定義顏色 (R, G, B) - 務必使用 Tuple () 而不是 List []
palette = [
    (220, 20, 60),   # pedestrian
    (119, 11, 32),   # rider
    (0, 0, 142),     # car
    (0, 0, 230),     # truck
    (106, 0, 228),   # bus
    (0, 60, 100),    # train
    (0, 80, 100),    # motorcycle
    (0, 0, 70),      # bicycle
    (0, 0, 192),     # traffic light
    (250, 170, 30)   # traffic sign
]

# 將類別與顏色包裝成 metainfo
metainfo = dict(classes=classes, palette=palette)

data_root = 'data/bdd100k_ninja/'

# --- 2. 修改模型頭部 ---
model = dict(
    bbox_head=dict(
        num_classes=10  # 改成 10 類
    ))

# --- 3. 資料載入設定 ---
train_dataloader = dict(
    batch_size=2,  
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,  # 使用包含 palette 的 metainfo
        ann_file='annotations/bdd100k_train.json',
        data_prefix=dict(img='train/img/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,  # 使用包含 palette 的 metainfo
        ann_file='annotations/bdd100k_val.json',
        data_prefix=dict(img='val/img/')))

test_dataloader = val_dataloader

# --- 4. 評估設定 ---
val_evaluator = dict(
    ann_file=data_root + 'annotations/bdd100k_val.json',
    metric='bbox')
test_evaluator = val_evaluator

# --- 5. 訓練時間 ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)

# 學習率
optim_wrapper = dict(optimizer=dict(lr=0.0001))

# 輸出位置
work_dir = './work_dirs/deformable_detr_bdd_ninja'

# --- 6. [重要] 視覺化設定 (確保可以存圖) ---
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')