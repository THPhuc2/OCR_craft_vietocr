import cv2
import numpy as np
import math
from PIL import Image, ImageOps
import torch

# 比率調整
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

img_path = "D:\\CODE\\Deep_Learning\\Project\\OCR_dbnet_craft_vietocr\\data\\test2\\\\0026.png"
img = Image.open(img_path)
if img.mode == 'RGBA':
    img = img.convert('RGB')

# Giả sử model_craft đã được khởi tạo
boxes = model_craft.get_boxes(img)

if isinstance(boxes, list) and len(boxes) > 0:
    boxes = boxes
    angle = calculate_rotation_angle(boxes)
    print(f"Góc xoay tính toán: {angle} độ")

    if angle <= 100:
        if 80 <= abs(angle) <= 100:
            angle = 90 if angle > 0 else -90
        else:
            angle = np.max([angle, -angle])
        rotated_image = rotate_image(img, angle)
    else:
        if 80 <= abs(angle) <= 100:
            angle = 90 if angle > 0 else -90
        else:
            angle = np.max([angle, -angle])
        rotated_image = rotate_image(img, -angle)

    print("hình ảnh đã được xoay")
else:
    print("Dữ liệu `boxes` không phải là danh sách hoặc danh sách rỗng.")

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

def get_min_max_pos(bbox):
    bbox = np.array(bbox)
    minx = int(min(bbox[:, 0]))
    maxx = int(max(bbox[:, 0]))
    miny = int(min(bbox[:, 1]))
    maxy = int(max(bbox[:, 1]))
    return minx, miny, maxx, maxy

boxes = model_craft.get_boxes(rotated_image)

min_x1, min_y1 = float('inf'), float('inf')
max_x2, max_y2 = float('-inf'), float('-inf')

for box in boxes:
    minx, miny, maxx, maxy = get_min_max_pos(box)
    min_x1 = min(min_x1, minx)
    min_y1 = min(min_y1, miny)
    max_x2 = max(max_x2, maxx)
    max_y2 = max(max_y2, maxy)

top_left = (min_x1, min_y1)
top_right = (max_x2, min_y1)
bottom_left = (min_x1, max_y2)
bottom_right = (max_x2, max_y2)

points = [top_left[0], top_left[1], top_right[0], top_right[1], bottom_right[0], bottom_right[1], bottom_left[0], bottom_left[1]]
# Convert rotated_image to a NumPy array before applying homography
rotated_image_np = np.array(rotated_image)

# Apply homography on the NumPy array version of the image
warped_image = homography(rotated_image_np, points)

img_save = 'D:\\CODE\\Deep_Learning\\Project\\OCR_dbnet_craft_vietocr\\data\\test2\\mask.jpg'
cv2.imwrite(img_save, warped_image)
print(f"Đã lưu ảnh phát hiện bounding box tại: {img_save}")
