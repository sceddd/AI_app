import json
import json
import logging
import pickle

import lmdb
import requests
from celery import shared_task, chain
from django.conf import settings
from django.shortcuts import get_object_or_404
from torchvision import transforms

from .app_models.photos import FaceEmbedding, get_photo_class, AbstractPhoto, OCRPhoto, ObjectDetPhoto

redis_client = settings.REDIS_CLIENT
mongo_client = settings.MONGO_CLIENT
lmdb_limit = settings.LMDB_LIMIT

logger = logging.getLogger(__name__)


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


@shared_task(queue='image_processing')
def process_batch(indices,function_type='face'):
    if function_type == 'face':
        return face_recognition_process(indices)
    elif function_type == 'ocr':
        return ocr_process(indices)
    elif function_type == 'ob_det':
        return det_process(indices)
    raise ValueError(f"Unknown function type: {function_type}")


def ocr_process(indices):
    err = []
    logger.info(f"Processing images {indices}")
    ocr_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})
    logger.info(f"Payload: {ocr_payload}")
    try:
        ocr_response = requests.post(settings.TORCHSERVE_URI_OCR, headers={'Content-Type': 'application/json'},
                                     data=ocr_payload)
        ocr_response.raise_for_status()
        ocr_response = ocr_response.json()
        if ocr_response == [[]]:
            err.append('OCR failed.')
            return [], [], err
        logger.info(f"OCR response: {ocr_response}")
        [process_ocr_image(ocr_data, err)for ocr_data in ocr_response]
    except requests.exceptions.RequestException as e:
        err.append(f"Error: {str(e)}")
    return err


def det_process(indices):
    err = []
    logger.info(f"Processing images {indices}")
    od_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})
    logger.info(f"Payload: {od_payload}")
    try:
        det_response = requests.post(settings.TORCHSERVE_URI_OD, headers={'Content-Type': 'application/json'},
                                     data=od_payload)
        det_response.raise_for_status()
        det_response = det_response.json()
        logger.info(f"OD response: {det_response}")
        [process_od_image(od_data, err)for od_data in det_response]
        return
    except requests.exceptions.RequestException as e:
        err.append(f"Error: {str(e)}")
    return err


def process_od_image(od_data, err):
    idx = od_data.get('idx')
    try:
        photo = get_object_or_404(ObjectDetPhoto, image_id=idx)
        photo.bounding_boxes= od_data.get('boxes', []) if od_data.get('boxes') else []
        photo.objects_det = od_data.get('objects', []) if od_data.get('objects') else []
        photo.status = AbstractPhoto.Status.RESULT_SAVED
        photo.save()
        logger.info(f"Image {photo} processed successfully.")
    except Exception as e:
        err.append(f"Error processing image {idx}: {e}")


def process_ocr_image(ocr_data, err):
    idx = ocr_data.get('idx')
    try:
        photo = get_object_or_404(OCRPhoto, image_id=idx)
        photo.texts = ocr_data.get('texts', []) if ocr_data.get('texts') else []
        photo.bounding_boxes = ocr_data.get('boxes', []) if ocr_data.get('boxes') else []
        photo.status = AbstractPhoto.Status.RESULT_SAVED
        photo.save()
        logger.info(f"Image {photo} processed successfully.")

    except Exception as e:
        err.append(f"Error processing image {idx}: {e}")


def face_recognition_process(indices):
    err = []
    logger.info(f"face recognition for images: {indices}")
    face_det_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})
    try:
        # Face detection
        face_det_response = detect_faces(face_det_payload)
        logger.info(f"Face detection response: {face_det_response}")
        faces_payload = json.dumps({
            'batch_results': face_det_response,
            'lmdb_path': settings.LMDB_PATH_FACE
        })

        if face_det_response == [[]]:
            err.append('Face detection failed.')
            return [], [], err
        # Face embedding
        face_embed = face_to_embed(faces_payload)
        if not face_embed:
            err.append('Face recognition failed.')
            return [], [], err
        # logger.info(f"Face recognition response: {type(face_embed)}")
        boxes = [face.get('boxes') for face in face_det_response]
        err = process_response(indices, boxes, face_embed, err, settings.LMDB_PATH_FACE,function_type='face')
    except requests.exceptions.RequestException as e:
        err.append(f"Error: {str(e)}")
    return err


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


def process_response(indices,boxes, reg_response, error, lmdb_path,function_type):
    face_env = lmdb.open(lmdb_path, readonly=True, lock=False)

    photo_type = get_photo_class(function_type)
    logger.info(f"Processing boxes {boxes}")
    for img_idx,boxes, idx in zip(indices, boxes,reg_response.keys()):
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
                    process_face_up_db(_img, reg_response[idx], idx)
                    logger.info(f"Saving face {idx} to DB")
                    results.append(idx)
                else:
                    error.append(f"Unexpected value in response: {idx}")
            photo.faces = results
            photo.bounding_boxes = boxes
            photo.status = AbstractPhoto.Status.RESULT_SAVED
            photo.save()

        except Exception as e:
            error.append(f"Error processing image {idx}:{e}")
    face_env.close()
    return error


# @upload_success
@shared_task(queue='image_upload')
def batch_upload_images_to_mongodb(image_ids,photo_type):
    cache = {}
    results = []
    for image_id in image_ids:
        try:
            photo_class = get_photo_class(photo_type)
            photo = get_object_or_404(photo_class,image_id=image_id)
            image_data = redis_client.get(image_id)
            if image_data:
                logger.info(f"Processing image {image_id}")
                photo.status = photo.Status.PROCESSING
                photo.save_image_to_gridfs(image_data)
                photo.save()
                logger.info(f"Image {photo.gridfs_id} saved to GridFS.")
            else:
                results.append(f"Image {image_id} not found in Redis.")
                continue

            if redis_client.exists(image_id):
                redis_client.delete(image_id)

            if image_data is None:
                results.append(f'Image {image_id} not found in Redis.')
                continue
            cache[image_id] = image_data
        except Exception as e:
            # Log the error
            logger.error(f"Error processing image {image_id}: {str(e)}")
            results.append(f"Error processing image {image_id}: {str(e)}")
    if cache:
        chain(
            write_cache_and_process.s(cache) |
            process_batch.s(function_type=photo_type)
        ).apply_async()
    return results


def write_cache(cache):
    env = lmdb.open(settings.LMDB_PATH, map_size=lmdb_limit)
    with env.begin(write=True) as txn:
        for key, value in cache.items():
            txn.put(key.encode('utf-8'), value)
    logger.info(f"Cache successfully written for {len(cache)} images.")


@shared_task(queue='write_cache_and_process')
def write_cache_and_process(cache):
    logger.info(f"Writing cache for {len(cache)} images.")
    if not cache or not isinstance(cache, dict):
        logger.error("Cache is empty or invalid")
        return "Cache is empty or invalid"
    try:
        write_cache(cache)
    except Exception as e:
        logger.error(f"Error writing cache: {str(e)}")
        return f"Error writing cache: {str(e)}"

    return list(cache.keys())
