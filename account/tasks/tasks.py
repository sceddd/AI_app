import json
import logging
from datetime import timedelta

import lmdb
from celery import shared_task, chain
from celery.result import AsyncResult
from celery.utils.time import timezone
from django.conf import settings
from django.shortcuts import get_object_or_404

from .face_task import face_recognition_process, pairwise_find
from .obdet import det_process
from .ocr_task import ocr_process
from ..app_models.photos import get_photo_class, FacePhoto, OCRPhoto, ObjectDetPhoto
from ..project_utils.utils import push_failed_task_id_to_ssd

logger = logging.getLogger(__name__)

redis_client = settings.REDIS_CLIENT
mongo_client = settings.MONGO_CLIENT
lmdb_limit = settings.LMDB_LIMIT
env = lmdb.open(settings.LMDB_PATH, map_size=lmdb_limit)

@shared_task
def update_is_new_status():
    one_hour_ago = timezone.now() - timedelta(hours=1)
    logger.info('Updating is_new status for photos older than 1 hour.')
    for model in [FacePhoto, OCRPhoto, ObjectDetPhoto]:
        model.objects.filter(is_new=True, created_at__lt=one_hour_ago).update(is_new=False)


@shared_task(queue='image_processing')
def process_batch(indices,function_type='face'):
    if function_type == 'face':
        return face_recognition_process.apply_async(args=[indices], queue='image_processing')
    elif function_type == 'ocr':
        return ocr_process.apply_async(args=[indices], queue='image_processing')
    elif function_type == 'ob_det':
        return det_process.apply_async(args=[indices], queue='image_processing')
    return ValueError(f"Unknown function type: {function_type}")


@shared_task(bind=True, queue='image_upload')
def batch_upload_images_to_mongodb(self, image_ids, photo_type):
    cache = {}
    error = []
    task_id = self.request.id
    photo_class = get_photo_class(photo_type)

    for image_id in image_ids:
        try:
            photo = get_object_or_404(photo_class, image_id=image_id)
            image_data = redis_client.get(image_id)
            if image_data:
                logger.info(f"Processing image {image_id}")
                photo.status = photo.Status.PROCESSING
                photo.save_image_to_gridfs(image_data)
                photo.save()
                logger.info(f"Image {photo.gridfs_id} saved to GridFS.")
            else:
                error.append(f"Image {image_id} not found in Redis.")
                continue

            if image_data is None:
                error.append(f'Image {image_id} not found in Redis.')
                continue
            cache[image_id] = image_data
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error processing image {image_id}: {error_message}")
            push_failed_task_id_to_ssd(task_id, image_ids=image_ids, photo_type=photo_type, error=error_message)
            return {'status':'failure','error':error_message}

    if cache:
        chain(
            write_cache_and_process.s(cache) |
            process_batch.s(function_type=photo_type)
        ).apply_async()

    return {'status': 'success', 'error': error}


def write_cache(cache):

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
def find_similar(faces_embed,face_id):
    pairwise_find(faces_embed,face_id)


@shared_task(queue='image_processing')
def restart_failed_tasks():
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        for task_id, task_info in cursor:
            task_id = task_id.decode('utf-8')
            task_info = json.loads(task_info.decode('utf-8'))

            error_message = task_info.get('error', '')
            process_batch.apply_async()
            if 'OCRFailed' in error_message:
                # Khởi động lại task `ocr_process`
                ocr_process.apply_async(
                    args=(task_info['indices'],)
                )
            elif 'DETFailed' in error_message:
                face_recognition_process.apply_async(
                    args=(task_info['indices'],)
                )
            elif 'REGFailed' in error_message:
                # Khởi động lại task `face_recognition_process` hoặc task khác liên quan đến REGFailed
                face_recognition_process.apply_async(
                    args=(task_info['indices'],)
                )
            elif 'UploadFailed' in error_message:
                # Khởi động lại task `batch_upload_images_to_mongodb`
                batch_upload_images_to_mongodb.apply_async(
                    args=(task_info['image_ids'], task_info['photo_type'])
                )

            # Xóa task khỏi LMDB sau khi xử lý
            txn.delete(task_id.encode('utf-8'))