import logging

import lmdb
from celery import shared_task, chain
from celery.result import AsyncResult
from django.conf import settings
from django.shortcuts import get_object_or_404

from .face_task import face_recognition_process, process_cluster
from .obdet import det_process
from .ocr_task import ocr_process
from ..app_models.photos import get_photo_class

logger = logging.getLogger(__name__)

redis_client = settings.REDIS_CLIENT
mongo_client = settings.MONGO_CLIENT
lmdb_limit = settings.LMDB_LIMIT


@shared_task
def task_status(task_id):
    result = AsyncResult(task_id)
    task_state = result.state
    task_info = result.info
    if task_state == 'SUCCESS':
        return result.result
    elif task_state.contains('FAILURE'):
        return str(result.info)
    else:
        return f"Task is {task_state} with info: {task_info}"


@shared_task(queue='image_processing')
def process_batch(indices,function_type='face'):
    if function_type == 'face':
        return face_recognition_process.apply_async(args=[indices], queue='image_processing')
    elif function_type == 'ocr':
        return ocr_process.apply_async(args=[indices], queue='image_processing')
    elif function_type == 'ob_det':
        return det_process.apply_async(args=[indices], queue='image_processing')
    return ValueError(f"Unknown function type: {function_type}")


@shared_task(bind=True,queue='image_upload')
def batch_upload_images_to_mongodb(self,image_ids,photo_type):
    cache = {}
    results = []
    photo_class = get_photo_class(photo_type)
    self.update_state(state='PROGRESS', meta={'image_id': image_ids, 'photo_type': photo_type})
    for image_id in image_ids:
        try:
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
            self.update_state(state='FAILURE', meta={'error': str(e)})
            logger.error(f"Error processing image {image_id}: {str(e)}")
            results.append(f"Error processing image {image_id}: {str(e)}")
            return results
    if cache:
        chain(
            write_cache_and_process.s(cache) |
            process_batch.s(function_type=photo_type)
        ).apply_async()
        self.update_state(state='SUCCESS', meta={'image_ids': image_ids, 'photo_type': photo_type})
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


@shared_task(queue='image_processing')
def cluster_face():
    process_cluster.apply_async()