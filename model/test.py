import json

import numpy as np
import torch
import ultralytics
from PIL import Image

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, (str, bytes)) or obj is None:
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def clean_json_dict(data):
    # Kiểm tra nếu 'objects' là một chuỗi JSON
    if isinstance(data['objects'], str):
        # Phân tích cú pháp chuỗi JSON thành đối tượng Python
        objects_list = json.loads(data['objects'])

        for obj in objects_list:
            # Định dạng lại 'conf' với 2 chữ số thập phân
            obj['conf'] = [f"{float(conf):.2f}" for conf in obj['conf']]

            # Đảm bảo 'class_ids' là dạng int
            obj['class_ids'] = [int(class_id) for class_id in obj['class_ids']]

        # Ghi đè lại 'objects' với đối tượng đã được làm sạch
        data['objects'] = objects_list
    data['boxes'] = [[[int(value) for value in box] for box in box_list] for box_list in data['boxes']]
    print(type(data))
    return data


model = ultralytics.YOLO("./object_detection/weight/yolov8n.pt")
im = Image.open('download.jpeg')
results = model.predict(im, conf=0.5)
objects = []
object_info =[{'conf': result.boxes.conf.tolist(),'class_ids': result.boxes.cls.tolist(),'classes': [model.names[int(cls_id)] for cls_id in result.boxes.cls.tolist()]
} for result in results
]
json_object = json.dumps(object_info, indent=4)
objects.append({'idx': "aaaaaaaaaaaa",
                'boxes': [ob_det.boxes.xyxy for ob_det in results],
                'objects': json_object
})
print(object_info)
print([clean_json_dict(x) for x in convert_to_serializable(objects)] )
# if results.tolist().boxes is not None:
#     object_info = {
#         'conf': results.boxes.conf.tolist(),  # Convert tensor to list
#         'class_ids': results.boxes.cls.tolist(),  # Convert tensor to list
#         'classes': [model.names[int(cls_id)] for cls_id in results.boxes.cls.tolist()]
#     }
#     objects.append(object_info)
# print(objects)