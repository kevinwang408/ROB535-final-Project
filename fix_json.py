import json
import os

# 定義要修復的檔案路徑
files_to_fix = [
    'data/bdd100k_ninja/annotations/bdd100k_train.json',
    'data/bdd100k_ninja/annotations/bdd100k_val.json'
]

# 定義要補進去的 info 內容 (這只是給程式看的 metadata，內容隨意)
dummy_info = {
    "description": "BDD100K Converted",
    "url": "",
    "version": "1.0",
    "year": 2025,
    "contributor": "Dataset Ninja",
    "date_created": "2025-11-23"
}

for file_path in files_to_fix:
    if os.path.exists(file_path):
        print(f"正在修復: {file_path} ...")
        
        # 1. 讀取現有的 JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 2. 檢查並補上 info 欄位
        if 'info' not in data:
            data['info'] = dummy_info
            print(" -> 已補上 'info' 欄位")
        else:
            print(" -> 'info' 欄位已存在，跳過")
            
        # 3. 寫回檔案
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print("✅ 完成！")
    else:
        print(f"❌ 找不到檔案: {file_path}")