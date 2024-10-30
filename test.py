# import lmdb
#
# import io
# import json
# import logging
#
# import lmdb
# import numpy as np
# import torch
# from PIL import Image
# from ultralytics import YOLO
#
#
# def convert_to_serializable(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, torch.Tensor):
#         print("tensor")
#         return obj.cpu().numpy().tolist()
#     elif isinstance(obj, dict):
#         return {k: convert_to_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         print(f"{obj}list")
#         return [convert_to_serializable(i) for i in obj]
#     elif isinstance(obj, tuple):
#         return tuple(convert_to_serializable(i) for i in obj)
#     elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, (str, bytes)) or obj is None:
#         return obj
#     else:
#         raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
#
#
# def clean_json_dict(data):
#     if isinstance(data['objects'], str):
#         objects_list = json.loads(data['objects'])
#
#         for obj in objects_list:
#             obj['conf'] = [f"{float(conf):.2f}" for conf in obj['conf']]
#             obj['boxes'] = [[int(value) for value in box] for box in obj['boxes']]
#         data['objects'] = objects_list
#     return data
# lmdb_env_read = lmdb.open('/home/victor-ho/work/school/final/backend/WODex/lmdb/user/1', readonly=True)
# batch = ["1_1000004674.jpg"]
# model = YOLO("/home/victor-ho/work/school/final/backend/WODex/model/object_detection/weights/yolov8s-world.pt")
# results = []
# with lmdb_env_read.begin(write=False) as txn:
#     for idx in batch:
#         try:
#             print(f"Processing image {idx}")
#             model.set_classes([])
#             image_data = txn.get(idx.encode('utf-8'))
#             image = Image.open(io.BytesIO(image_data))
#             if image.mode == 'RGBA':
#                 image = image.convert('RGB')
#             ob_dets = model.predict(image, conf=0.1)
#             objects = [{
#                     'conf': ob_det.boxes.conf.tolist(),
#                     'classes': [model.names[int(cls_id)] for cls_id in ob_det.boxes.cls.tolist()],
#                     'boxes': ob_det.boxes.xyxy.tolist(),
#                 }
#                 for ob_det in ob_dets
#             ]
#
#             json_object = json.dumps(objects, indent=4)
#             results.append({
#                 'idx': idx,
#                 'objects': json_object
#             })
#         except Exception as e:
#             logging.error(f"Detect failed for image: {e}")
#             results.append({
#                 'idx': idx,
#                 'boxes': None,
#                 'objects': None
#             })
#
# # print(results)
# inference_output = convert_to_serializable(results)
# output = [clean_json_dict(f) for f in inference_output]
# print(output)
