import io
import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

import lmdb
import requests
from celery import shared_task
from django.conf import settings
from gridfs import GridFS
from torchvision import transforms

from .app_models.photos import FaceEmbedding, Photo

redis_client = settings.REDIS_CLIENT
mongo_client = settings.MONGO_CLIENT
logger = logging.getLogger(__name__)


def process_image_up_db(image_data, embed, idx):
    db = mongo_client['user_images']
    fs = GridFS(db)
    if isinstance(image_data, bytes):
        with io.BytesIO(image_data) as f:
            logger.info(f"Saving face image {type(image_data)} to GridFS")

            file_id = fs.put(f, filename=f"face_{idx}.jpg")
            face_embedding = FaceEmbedding(
                face_id=idx,
                embedding=embed,
                gridfs_id=str(file_id)
            )
            face_embedding.save_image(image_data)
            face_embedding.save()
            return face_embedding
    else:
        raise ValueError("Image data must be bytes.")


def process_batch(indices):
    err = []
    indices = check_lmdb_indices(indices)
    if not indices:
        err.append('No valid indices to process.')
        logger.warning("No valid indices found to process.")
        return [], []

    face_det_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})

    try:
        # Face detection
        face_det_response = detect_faces(face_det_payload)

        faces_payload = json.dumps({
            'batch_results': face_det_response,
            'lmdb_path': settings.LMDB_PATH_FACE
        })

        # Face embedding
        face_embed = face_to_embed(faces_payload)
        if not(face_det_response and face_embed):
            err.append('Face recognition failed.')
            return [], [], err

        err = process_response(indices, face_embed, err)
    except requests.exceptions.RequestException as e:
        err.append(f"Error: {str(e)}")
        face_det_response = [f'Error: {str(e)}'] * len(indices)
        face_embed = [f'Error: {str(e)}'] * len(indices)

    return face_det_response,face_embed, err


def check_lmdb_indices(indices):
    env = lmdb.open(settings.LMDB_PATH, readonly=True, lock=False)
    db = env.open_db()
    valid_index = []
    with env.begin(write=False) as txn:
        with txn.cursor(db) as cursor:
            for idx in indices:
                key = idx.encode('utf-8')
                if cursor.set_key(key):
                    valid_index.append(idx)
                else:
                    logger.warning(f"Index {idx} not found in LMDB")
    return valid_index


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


def process_response(indices,reg_response,error):
    face_env = lmdb.open(settings.LMDB_PATH_FACE, readonly=True, lock=False)
    # logger.info(f"Processing response for {reg_response} images")

    for img_idx, face_idx in zip(indices, reg_response.keys()):
        face_embeds = []
        photo = Photo.objects.get(image_id=img_idx)
        logger.info(f"Processing image {face_idx}")
        
        with face_env.begin() as txn:
            if isinstance(face_idx, str) and (reg_response[face_idx] is not None):
                stored_data = txn.get(face_idx.encode('utf-8'))
                logger.info(f"Processing face {face_idx}")
                if stored_data is None:
                    error.append(f"No data found for image with idx {face_idx} in LMDB")
                    continue
                face_data = pickle.loads(stored_data).get("face")
                face_img = transforms.ToPILImage()(face_data).tobytes()
                face_embed = process_image_up_db(face_img,reg_response[face_idx],face_idx)
                face_embeds.append(face_embed)
            else:
                logger.warning(f"Unexpected value in response: {face_idx}")
        photo.faces = face_embeds
        photo.status = Photo.Status.EMBEDDING_SAVED
        photo.save()
    face_env.close()
    return error


# @upload_success
@shared_task
def batch_upload_images_to_mongodb(image_ids):
    cache = {}
    results = []
    results_queue = Queue()

    with ThreadPoolExecutor(max_workers=3) as executor:
        def handle_results(future: Future):
            if future.result() is None:
                pass
            else:
                results.extend(future.result())

        for image_id in image_ids:
            photo = Photo.objects.get(image_id=image_id)
            image_data = redis_client.get(image_id)
            photo.status = Photo.Status.PROCESSING
            photo.save()

            if redis_client.exists(image_id):
                redis_client.delete(image_id)

            if image_data is None:
                results.append(f'Image {image_id} not found in Redis.')
                continue
            cache[image_id] = image_data

            if len(cache) >= settings.LMDB_BATCH_SIZE:
                future = executor.submit(write_cache_and_process, cache, results_queue)
                future.add_done_callback(handle_results)
                cache = {}

        if cache:
            future = executor.submit(write_cache_and_process, cache, results_queue)
            future.add_done_callback(handle_results)

        executor.shutdown(wait=True)

        while not results_queue.empty():
            results.extend(results_queue.get())

        return results


def write_cache(cache):
    env = lmdb.open(settings.LMDB_PATH, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for key, value in cache.items():
            txn.put(key.encode('utf-8'), value)


def write_cache_and_process(cache, results_queue):
    write_cache(cache)
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_batch, cache.keys())]

        for future in futures:
            results.append(future.result())
    results_queue.put(results)
