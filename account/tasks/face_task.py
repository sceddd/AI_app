import json
import logging
import pickle

import lmdb
import numpy as np
import requests
from celery import shared_task
from django.conf import settings
from django.shortcuts import get_object_or_404
from sklearn.metrics import pairwise_distances
from torchvision import transforms
from ..project_utils.utils import push_failed_task_id_to_ssd, publish_new_results

from ..app_models.photos import get_photo_class, FaceEmbedding, AbstractPhoto

r = settings.REDIS_CLIENT
logger = logging.getLogger(__name__)


@shared_task(queue='image_processing')
def process_response(indices, boxes, reg_response, error, lmdb_path,function_type,task_id):
    face_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    cluster = settings.CLUSTER_MODEL
    photo_type = get_photo_class(function_type)
    logger.info(f"Processing boxes {boxes}")
    embeds = []
    for img_idx, boxes, idx in zip(indices, boxes, reg_response.keys()):
        try:
            results = []
            photo = get_object_or_404(photo_type, image_id=img_idx)
            with face_env.begin() as txn:
                if isinstance(idx, str) and (reg_response[idx] is not None):
                    stored_data = txn.get(idx.encode('utf-8'))
                    if stored_data is None:
                        error.append(f"No data found for image with idx {idx} in LMDB")
                        continue
                    _data = pickle.loads(stored_data).get("face")
                    _img = transforms.ToPILImage()(_data).tobytes()
                    _dr_embeds = cluster.to_latent_space(reg_response[idx],transform_only=True).squeeze().tolist()
                    process_face_up_db(_img, _dr_embeds, idx)
                    embeds.append(_dr_embeds)
                    logger.info(f"Saving face {idx} to DB")
                    results.append(idx)
                else:
                    error.append(f"Unexpected value in response: {idx}")
            photo.faces = results
            publish_new_results(img_idx, faces=results)
            photo.bounding_boxes = boxes
            photo.status = AbstractPhoto.Status.RESULT_SAVED
            photo.save()
        except Exception as e:
            logger.error(f"Error processing image {idx}:{e}")
            error_message = "face_SAVEFailed:{}".format(e)
            push_failed_task_id_to_ssd(task_id, indices=indices, error=error_message)
    face_env.close()

    return error


@shared_task(queue='image_processing')
def pairwise_find(embeds,new_point,k=4):
    logger.info(f"{'*'*100}")
    logger.info(embeds)
    logger.info(new_point)
    logger.info(f"{'*'*100}")
    distances = pairwise_distances(embeds, [new_point], metric='euclidean').flatten()
    nearest_indices = np.argsort(distances)[:k]
    return embeds[nearest_indices]


def process_face_up_db(image_data, embed, idx):
    try:
        logger.info(f"Saving image {idx} to DB")
        if isinstance(image_data, bytes):
            face_embedding = FaceEmbedding(
                face_id=idx,
                embedding=embed,
            )
            face_embedding.save_image(image_data)
            face_embedding.save()
        else:
            raise ValueError("Image data must be bytes.")
    except Exception as e:
        logger.error(f"Error saving image to GridFS: {str(e)}")
        return None


@shared_task(bind=True, queue='image_processing')
def face_recognition_process(self, indices):
    err = []
    task_id = self.request.id

    logger.info(f"face recognition for images: {indices}")
    face_det_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})

    try:
        # Face detection
        face_det_response = detect_faces(face_det_payload)
    except requests.exceptions.RequestException as e:
        error_message = f'face_DETFailed:{e}'
        push_failed_task_id_to_ssd(task_id, indices=indices, error=error_message)

        return {'status': 'failure', 'error': err}

    try:
        faces_payload = json.dumps({
            'batch_results': face_det_response,
            'lmdb_path': settings.LMDB_PATH_FACE
        })
        # Face embedding
        face_embed = face_to_embed(faces_payload)
        boxes = [face.get('boxes') for face in face_det_response]

        err = process_response(indices, boxes, face_embed, err, settings.LMDB_PATH_FACE,
                               function_type='face',
                               task_id=task_id)
    except requests.exceptions.RequestException as e:
        error_message = f'face_REGFailed:{e}'
        logger.error(f"Face Embedding Failed: {error_message}")
        push_failed_task_id_to_ssd(task_id, indices=indices, error=error_message)

        return {'status': 'failure', 'error': err}

    return {'status': 'success', 'error': err}


def detect_faces(image_data):
    det_response = requests.post(settings.TORCHSERVE_URI_DET, headers={'Content-Type': 'application/json'},
                                 data=image_data)
    det_response.raise_for_status()
    face_det_response = det_response.json().get('batch_results')
    return face_det_response


def face_to_embed(faces_payload):
    reg_response = requests.post(settings.TORCHSERVE_URI_REG, headers={'Content-Type': 'application/json'},
                                 data=faces_payload)
    reg_response.raise_for_status()
    face_reg_response = reg_response.json()
    return face_reg_response


