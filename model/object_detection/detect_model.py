import io
import json
import logging

import lmdb
import numpy as np
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO

logging.getLogger(__name__)


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
    if isinstance(data['objects'], str):
        objects_list = json.loads(data['objects'])

        for obj in objects_list:
            obj['conf'] = [f"{float(conf):.2f}" for conf in obj['conf']]

            obj['class_ids'] = [int(class_id) for class_id in obj['class_ids']]

        data['objects'] = objects_list
    data['boxes'] = [[[int(value) for value in box] for box in box_list] for box_list in data['boxes']]
    return data


class OCRHandler(BaseHandler):
    def initialize(self, context):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO("yolov8n.pt")

    def preprocess(self, data):
        request = data[0].get('body')
        if isinstance(request, dict):
            payload = request
        else:
            payload = json.loads(request)
        input_path = payload.get("lmdb_path", None)
        if input_path is None:
            raise ValueError(f"lmdb_path must be provided in the payload and cannot be None")
        self.lmdb_env_read = lmdb.open(input_path, readonly=False, lock=False)
        return payload.get('idx')

    def inference(self, batch, *args, **kwargs):
        results = []
        logging.info("Input: {}".format(batch))
        logging.info(f"Detect object in {len(batch)} images")

        with self.lmdb_env_read.begin(write=False) as txn:
            for idx in batch:
                try:
                    logging.info(f"Processing image {idx}")
                    image_data = txn.get(idx.encode('utf-8'))
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    ob_dets = self.model.predict(image, conf=0.5)

                    objects = [{
                        'conf': ob_det.boxes.conf.tolist(),
                        'class_ids': ob_det.boxes.cls.tolist(),
                        'classes': [self.model.names[int(cls_id)] for cls_id in ob_det.boxes.cls.tolist()]
                        }
                        for ob_det in ob_dets
                    ]

                    json_object = json.dumps(objects, indent=4)
                    results.append({
                        'idx': idx,
                        'boxes': [ob_det.boxes.xyxy for ob_det in ob_dets],
                        'objects': json_object
                    })
                except Exception as e:
                    logging.error(f"Detect failed for image: {e}")
                    results.append({
                        'idx': idx,
                        'boxes': None,
                        'objects': None
                    })
        return results

    def postprocess(self, inference_output):
        inference_output = convert_to_serializable(inference_output)
        output = [clean_json_dict(f) for f in inference_output]
        return [output]
