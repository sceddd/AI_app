import numpy as np
import cv2
import torchvision.transforms as transforms
import yaml


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def compute_center_y(box):
    y_coords = [point[1] for point in box]
    return (min(y_coords) + max(y_coords)) / 2


def compute_height(box):
    y_coords = [point[1] for point in box]
    return max(y_coords) - min(y_coords)


def is_on_same_line(box_a, box_b, max_y_distance):
    center_a = compute_center_y(box_a)
    center_b = compute_center_y(box_b)
    distance = abs(center_a - center_b)
    return distance <= max_y_distance


def sort_by_line(box_info, threshold=0.5):
    box_info.sort(key=lambda box: compute_center_y(box))

    all_same_row = []
    same_row = [box_info[0]]
    average_height = compute_height(box_info[0])

    for i in range(len(box_info) - 1):
        box_a = box_info[i]
        box_b = box_info[i + 1]
        height_b = compute_height(box_b)
        average_height = (average_height * len(same_row) + height_b) / (len(same_row) + 1)
        max_y_distance = average_height * threshold

        if is_on_same_line(box_a, box_b, max_y_distance):
            same_row.append(box_b)
        else:
            all_same_row.append(same_row)
            same_row = [box_b]
            average_height = compute_height(box_b)

    if same_row:
        all_same_row.append(same_row)

    sorted_same_row = []
    for same in all_same_row:
        same.sort(key=lambda box: min([point[0] for point in box]))
        sorted_same_row.append(same)

    return sorted_same_row


def test_preprocess(img,
                    new_size=736,
                    pad=False):
    img = test_resize(img, size=new_size, pad=pad)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)
    return img


def test_resize(img, size=736, pad=False):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    if pad:
        new_img = np.zeros((size, size, c), img.dtype)
        new_img[:h, :w] = cv2.resize(img, (w, h))
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img


def read_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def draw_bbox(img, result, color=(255, 0, 0), thickness=2):
    """
    :input: RGB img
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    for point in result:
        point = np.array(point).astype(int)
        point = point.reshape((-1, 1, 2))
        cv2.polylines(img, [point], True, color, thickness)
    return img
