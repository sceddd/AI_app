import cv2


def draw_bounding_boxes_opencv(image, boxes, output_path):
    """
    Vẽ các bounding box lên hình ảnh sử dụng OpenCV.

    Args:
        image (numpy array): Hình ảnh gốc.
        boxes (list of tuples): Danh sách các bounding box, mỗi box là một tuple (x_min, y_min, x_max, y_max).
        output_path (str): Đường dẫn để lưu hình ảnh sau khi vẽ.

    Returns:
        None
    """
    for box in boxes:
        x_min, y_min, x_max, y_max = list(map(int, box))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=3)

    cv2.imwrite(output_path, image)

