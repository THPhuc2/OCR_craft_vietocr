import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from CRAFT import CRAFTModel, draw_boxes, draw_polygons, boxes_area, polygons_area
import gc
from VietOCR_model import VietOCR_model


def model_craft(cache_dir='weights/', device='cuda:0', use_refiner=True, fp16=False):
    return CRAFTModel(cache_dir=cache_dir, device=device, use_refiner=use_refiner, fp16=fp16)

def get_detected_bbox_craft(image, model_craft, device="cpu"):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        bboxes = model_craft.get_boxes(image)
    return bboxes


def draw_bbox(img, bboxes, color=(255, 0, 0), thickness=2):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        minx, miny = int(min(bbox[:, 0])), int(min(bbox[:, 1]))
        maxx, maxy = int(max(bbox[:, 0])), int(max(bbox[:, 1]))
        draw.rectangle([minx, miny, maxx, maxy], outline=color, width=thickness)
    return img


def get_min_max_pos(bbox):
    bbox = np.array(bbox)
    minx = int(min(bbox[:, 0]))
    maxx = int(max(bbox[:, 0]))
    miny = int(min(bbox[:, 1]))
    maxy = int(max(bbox[:, 1]))

    return minx, miny, maxx, maxy


def get_cropped_area(image, bbox):
    minx, miny, maxx, maxy = get_min_max_pos(bbox)
    return image.crop((minx, miny, maxx, maxy))

def recog_and_draw(image, bboxes, model_ocr, font_path="D:\\CODE\\Deep_Learning\\Project\\OCR_dbnet_craft_vietocr\\vietocr\\Arial.ttf", device="cuda:0"):  # Roboto-Regular.ttf
    torch.cuda.empty_cache()
    gc.collect()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    try:
        font = ImageFont.truetype(font_path, 16)
    except IOError:
        print(f"Không tìm thấy font tại {font_path}. Sử dụng font mặc định.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        cropped_image = get_cropped_area(image, bbox)
# Kiểm tra kích thước ảnh bị cắt
        if cropped_image.size[0] == 0 or cropped_image.size[1] == 0:
            print(f"Warning: Bounding box {bbox} cắt ra ảnh có kích thước không hợp lệ. Bỏ qua box này.")
            continue
        # pred_text = model_ocr.predict(cropped_image)
                # Thực hiện nhận diện văn bản
        try:
            pred_text = model_ocr.predict(cropped_image)
        except Exception as e:
            print(f"Lỗi khi nhận diện văn bản: {e}. Bỏ qua box {bbox}.")
            continue

        minx, miny, maxx, maxy = get_min_max_pos(bbox)


        draw.rectangle([minx, miny, maxx, maxy], outline="red", width=2)
        draw.text((minx, miny - 10), pred_text, fill="blue", font=font)

    return image


def create_result(image, model_craft, VietOCR_model, device="cuda:0"):

    bboxes = get_detected_bbox_craft(image, model_craft, device=device)


    result_image = recog_and_draw(image, bboxes, VietOCR_model, device=device)

    return result_image