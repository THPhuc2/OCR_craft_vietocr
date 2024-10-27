import os
import cv2
from datetime import datetime
from VietOCR_model import load_vietocr
from Craft_model import model_craft
from extract_table import recognize_structure, extract_and_save_tables
from rotation import recognize_text_in_folder
from utils import reassemble_tables_with_recognized_text_to_img


def main(df):
    # Chuẩn hóa đường dẫn để tương thích trên mọi hệ điều hành
    df = os.path.normpath(df)

    # Load mô hình OCR
    model_ocr = load_vietocr(device="cuda:0")

    # Lặp qua từng file trong thư mục
    for file_name in os.listdir(df):
        if file_name.endswith(".png"):
            # Đọc và chuyển đổi màu của ảnh
            bordered_table = cv2.imread(os.path.join(df, file_name))
            bordered_table = cv2.cvtColor(bordered_table, cv2.COLOR_BGR2RGB)

            # Tạo thư mục lưu kết quả
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_save_dir = os.path.join(df, f"test_resize_{os.path.splitext(file_name)[0]}_{timestamp}")
            if not os.path.exists(base_save_dir):
                os.makedirs(base_save_dir)

            # Khởi tạo danh sách lưu bảng
            list_table_boxes = []
            table_list = [bordered_table]

            # Xử lý từng bảng trong danh sách
            for table in table_list:
                finalboxes, output_img = recognize_structure(table, base_save_dir)
                list_table_boxes.append(finalboxes)
                extract_and_save_tables(bordered_table, finalboxes, base_save_dir)

            # Nhận dạng văn bản trong bảng
            recognize_text_in_folder(base_save_dir, model_craft, model_ocr, base_save_dir)

            # Kết hợp văn bản đã nhận dạng vào hình ảnh đầu ra
            reassemble_tables_with_recognized_text_to_img(base_save_dir, base_save_dir)


# Gọi hàm main với đường dẫn df do người dùng nhập
if __name__ == "__main__":
    df = input("Nhập đường dẫn thư mục chứa ảnh (có thể dùng dấu / hoặc \\) ảnh là .png không nhập ảnh khác: ")
    main(df)
