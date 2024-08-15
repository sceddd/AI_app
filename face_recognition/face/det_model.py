import io
import json
import logging
import os
import pickle

import lmdb
import numpy as np
import torch
from PIL import Image
from bson import ObjectId
from facenet_pytorch import extract_face
from facenet_pytorch.models.mtcnn import MTCNN
from ts.torch_handler.base_handler import BaseHandler


def cal_wh(box):
    """
    Calculate the width and height of a bounding box.

    Parameters:
    - box: List of bounding box coordinates [x_min, y_min, x_max, y_max]

    Returns:
    - (width, height): Tuple of width and height of the bounding box
    """
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return width, height


def calculate_flm_on_cropped_img(box, f_lm, imgcr_size=160):
    tl = box[:2]
    w, h = cal_wh(box)
    new_pt = []
    for p in f_lm:
        new_x = (p[0] - tl[0]) * (imgcr_size / w)
        new_y = (p[1] - tl[1]) * (imgcr_size / h)
        new_pt.append([new_x, new_y])
    return new_pt


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, (str, bytes)) or obj is None:
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


logging.getLogger(__name__)


class FaceDetectionHandler(BaseHandler):
    def initialize(self, ctx):
        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.model = MTCNN(
            image_size=(160,160), margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device, keep_all=True
        )

        self.model.eval()

    def preprocess(self, data):
        request = data[0].get('body')
        logging.info(request)

        if isinstance(request, dict):
            payload = request
        else:
            payload = json.loads(request)

        input_path = payload.get("lmdb_path",None)
        if input_path is None:
            raise ValueError(f"lmdb_path must be provided in the payload and cannot be None")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"LMDB path {input_path} does not exist.")

        output_path = os.path.join(input_path,'face', 'det')

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"LMDB path {output_path} does not exist.")

        self.lmdb_env_read = lmdb.open(input_path, readonly=False, lock=False)
        self.lmdb_env_write = lmdb.open(output_path, readonly=False, lock=False)

        return payload.get("idx")

    def inference(self, data, *args, **kwargs):
        results = []
        logging.info(f"Detecting faces in {len(data)} images")
        with self.lmdb_env_read.begin(write=False) as txn:
            for idx in data:
                try:
                    logging.info(f"Processing image {idx}")
                    image_data = txn.get(idx.encode('utf-8'))
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    face_data = self.model.detect(image, landmarks=True)
                    results.append((idx, image, face_data))
                except Exception as e:
                    logging.error(f"Face detection failed for image: {e}")
                    results.append((idx, image, None))

        return results

    def postprocess(self, data):
        batch_results = []
        logging.info(f"Postprocessing {data} images")

        with self.lmdb_env_write.begin(write=True) as txn:
            for item in data:
                img_idx, image, face_datas = item

                img_results = []
                boxes, confidences, landmarks = face_datas

                if boxes is None:
                    logging.info(f"No faces detected in image with idx {img_idx}.")
                    img_results.append(None)
                else:
                    for face_idx, (box, conf, lm) in enumerate(zip(boxes, confidences, landmarks)):
                        if conf > 0.9:
                            new_pt = calculate_flm_on_cropped_img(box, lm, imgcr_size=160)
                            face = extract_face(image, box)
                            face_key = str(ObjectId())

                            face_data = {
                                "face": face,
                                "new_pt": new_pt
                            }
                            txn.put(face_key.encode('utf-8'), pickle.dumps(face_data))
                            img_results.append(face_key)

                        else:
                            logging.info(f"Face detection confidence too low: {conf}")
                batch_results.append(img_results)
        logging.info(f"Postprocessing complete. Returning {batch_results} results.")
        result_with_lmdb_path = {
            "batch_results": batch_results
        }
        result_with_lmdb_path = convert_to_serializable(result_with_lmdb_path)
        return [result_with_lmdb_path]


_service = FaceDetectionHandler()
