import os
import cv2
import numpy as np


def recognize_text_in_folder(folder_path, model_craft, model_ocr, result_save_dir):
    """
    Thực hiện nhận diện văn bản cho tất cả các bảng trong folder và lưu kết quả.
    """

    print(folder_path)
    print(result_save_dir)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print("đọc được file")
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Đang xử lý file: {filename}")

            image = cv2.imread(file_path)
            if image is None:
                print(f"Không thể đọc file: {filename}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            result_image = create_result(image, model_craft, model_ocr, device="cuda:0")  # Gọi hàm nhận diện văn bản


            if isinstance(result_image, Image.Image):
                # Chuyển đổi từ PIL.Image sang NumPy
                result_image = np.array(result_image)


            if filename == "img.jpg":
                result_filename = "img_recognized.jpg"
            else:
                result_filename = filename.replace("table_", "result_table_")


            result_path = os.path.join(result_save_dir, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"Đã lưu kết quả nhận diện: {result_path}")