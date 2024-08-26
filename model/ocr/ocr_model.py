import json
import logging
import os
import pickle
import zipfile

import lmdb
import numpy as np
import torch
import yaml
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io

from ocr import MyOCR

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


class OCRHandler(BaseHandler):
    def initialize(self, context):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MyOCR(det_model='DB', reg_model='MASTER')
        pass

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
        logging.info(f"Extracting texts in {len(batch)} images")
        with self.lmdb_env_read.begin(write=False) as txn:
            for idx in batch:
                try:
                    logging.info(f"Processing image {idx}")
                    image_data = txn.get(idx.encode('utf-8'))
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    ocr = self.model.get_result(image)
                    results.append({
                        'idx': idx,
                        'boxes': ocr['boxes'],
                        'texts': ocr['texts']
                    })

                except Exception as e:
                    logging.error(f"OCR failed for image: {e}")
                    results.append({
                        'idx': idx,
                        'boxes': None,
                        'texts': None
                    })
        return results

    def postprocess(self, inference_output):
        inference_output = convert_to_serializable(inference_output)
        return [inference_output]
