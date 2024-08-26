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
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, (str, bytes)) or obj is None:
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def unzip_file(path):
    zip_path = os.path.join(path, "util.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)


class FaceRecognitionHandler(BaseHandler):
    def initialize(self, context):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_dir = context.system_properties.get('model_dir')
        unzip_file(model_dir)
        from util.inference import FaceRecognition

        recog_config = os.path.join(model_dir, 'util', 'configs','recognition_model.yaml')

        with open(recog_config, 'r') as file:
            self.cfg = yaml.safe_load(file)

        self.model = FaceRecognition(self.cfg['VGGFace'], self.device)
        logging.info("Face recognition model loaded successfully")

    def preprocess(self, data):
        logging.info(f"Received input: {data}")
        request = data[0].get('body')
        logging.info(request)

        if isinstance(request, dict):
            payload = request
        else:
            payload = json.loads(request)
        logging.info(f"Payload: {payload}")
        output_path = payload.get("lmdb_path", None)

        self.lmdb_env = lmdb.open(output_path, readonly=True, lock=False)

        if output_path is None:
            raise ValueError(f"lmdb_path must be provided in the payload and cannot be None")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"LMDB path {output_path} does not exist.")

        return payload.get("batch_results")

    def inference(self, batch, *args, **kwargs):
        results = {}
        for image in batch:
            for faces_idx in image['face_key']:
                try:
                    if faces_idx is None:
                        results[faces_idx] = None
                        continue
                    logging.info(f"Processing faces {faces_idx}")

                    with self.lmdb_env.begin() as txn:
                        stored_data = txn.get(faces_idx.encode('utf-8'))
                        if stored_data is None:
                            logging.error(f"No data found for image with idx {faces_idx}")
                            results[faces_idx] = None
                            continue

                        face_data = pickle.loads(stored_data)

                        new_pt = face_data.get("new_pt")

                        embedding = self.model.embed(face_data.get("face"), new_pt)
                        embedding = convert_to_serializable(embedding)
                        results[faces_idx] = embedding

                except Exception as e:
                    logging.error(f"Face recognition failed for image: {e}")
        return results

    def postprocess(self, inference_output):
        return [inference_output]


