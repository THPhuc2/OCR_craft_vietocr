import os
import cv2

def reassemble_tables_with_recognized_text_to_img(img_folder_path, table_folder_path):
    coords_file = os.path.join(table_folder_path, "table_coords.txt")
    if not os.path.exists(coords_file):
        print(f"Không tìm thấy file {coords_file}")
        return

    img_path = os.path.join(img_folder_path, "img_no_tables.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print("Không thể tải ảnh img_no_tables.jpg")
        return

    with open(coords_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        table_file, x, y, w, h = line.split()
        x, y, w, h = int(x), int(y), int(w), int(h)
        result_table_path = os.path.join(table_folder_path, table_file.replace("table_", "result_table_"))
        result_table_img = cv2.imread(result_table_path)
        if result_table_img is not None:
            img[y:y + h, x:x + w] = result_table_img
            print(f"Ghép lại bảng {result_table_path} vào vị trí ({x}, {y}, {w}, {h}) trên ảnh img.jpg")

    result_img_path = os.path.join(img_folder_path, "final_img_with_tables.jpg")
    cv2.imwrite(result_img_path, img)
    print(f"Ảnh img_recognized.jpg đã được ghép lại và lưu tại: {result_img_path}")

import numpy as np
import math
from PIL import Image

# Tỉ lệ điều chỉnh
W_RATIO = 1.0

def calculate_rotation_angle(bboxes):
    angles = []
    for bbox in bboxes:
        points = np.array(bbox).reshape(-1, 2)
        angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
        angles.append(angle)
        print(angle)

    max_angle = max(angles, key=abs)  # Lấy góc có giá trị tuyệt đối lớn nhất
    print(angles)
    print(max_angle)
    return max_angle

def rotate_image(image, angle):
    center = tuple(np.array(image.size) / 2)
    rotated = image.rotate(angle, resample=Image.BICUBIC, center=center, expand=True)
    return rotated

def homography(img, pin):
    p1 = np.array([pin[0], pin[1]])
    p2 = np.array([pin[2], pin[3]])
    p3 = np.array([pin[4], pin[5]])
    p4 = np.array([pin[6], pin[7]])

    o_width = np.linalg.norm(p2 - p1)
    o_width = math.floor(o_width * W_RATIO)

    o_height = np.linalg.norm(p3 - p2)
    o_height = math.floor(o_height)

    src = np.float32([p1, p2, p3, p4])
    dst = np.float32([[0, 0], [o_width, 0], [o_width, o_height], [0, o_height]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (o_width, o_height))