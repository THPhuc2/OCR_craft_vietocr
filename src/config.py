# config.py
import os
from datetime import datetime

# Tạo thư mục lưu trữ với timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_save_dir = f"/content/drive/MyDrive/code/ocr/Multi_Type_TD_TSR/output/test_table/test_{timestamp}"

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)
