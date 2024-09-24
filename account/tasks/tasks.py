import io
import json
import logging
import os
import zipfile
from datetime import timedelta
from pathlib import Path

import lmdb
from celery import shared_task, chain
from django.conf import settings
from django.utils import timezone

from .face_task import face_recognition_process, pairwise_find
from .obdet import det_process
from .ocr_task import ocr_process
from ..app_models.photos import FacePhoto, OCRPhoto, ObjectDetPhoto, AbstractPhoto
from ..models import CustomUser
from ..project_utils.account_utils import chunked_iterable
from ..project_utils.utils import push_failed_task_id_to_ssd, get_photo

logger = logging.getLogger(__name__)

redis_client = settings.REDIS_CLIENT
mongo_client = settings.MONGO_CLIENT
lmdb_limit = settings.LMDB_LIMIT


@shared_task
def system_refresh():
    update_status()
    refresh_lmdb()


@shared_task
def refresh_lmdb():
    #remove the lmdb folders settings.LMDB_PATH only
    for folder in os.listdir(settings.LMDB_PATH):
        if folder != 'user':
            path = Path(settings.LMDB_PATH) / folder
            for file in os.listdir(path):
                os.remove(path / file)


@shared_task
def update_status():
    one_hour_ago = timezone.now() - timedelta(hours=1)
    logger.info('Updating is_new status for photos older than 1 hour.')
    for model in [FacePhoto, OCRPhoto, ObjectDetPhoto]:
        [img.switch_status().save() for img in model.objects.filter(is_new=True, created_at__lt=one_hour_ago)]


@shared_task(queue='image_processing')
def process_batch(indices, function_type='face'):
    logger.info(f"Processing batch of {len(indices)} images for {function_type}")
    if function_type.startswith('face'):
        return face_recognition_process.apply_async(args=[indices], queue='image_processing')
    elif function_type.startswith('ocr'):
        return ocr_process.apply_async(args=[indices], queue='image_processing')
    elif function_type.startswith('ob_det'):
        pass
    return ValueError(f"Unknown function type: {function_type}")


def write_cache(cache, user_id):
    path = Path(settings.LMDB_PATH) / "user" / f"{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    user_dir = str(path)
    user_env = lmdb.open(user_dir, map_size= lmdb_limit * 50)  # 50MB
    with user_env.begin(write=True) as txn:
        for key, value in cache.items():
            txn.put(key.encode('utf-8'), value)
    logger.info(f"Cache successfully written for {len(cache)} images.")
    return True


@shared_task(bind=True, queue='image_processing')
def process_ob_det(indices, input_words):
    return det_process.apply_async(args=[indices, input_words], queue='image_processing')


@shared_task(queue='write_cache_and_process')
def write_cache_and_process(cache, user_id):
    logger.info(f"Writing cache for {len(cache)} images.")
    if not cache or not isinstance(cache, dict):
        logger.error("Cache is empty or invalid")
        return "Cache is empty or invalid"
    try:
        write_cache(cache, user_id)
    except Exception as e:
        logger.error(f"Error writing cache: {str(e)}")
        return f"Error writing cache: {str(e)}"

    return list(cache.keys())


@shared_task(bind=True, queue='image_upload')
def process_zip_file_lmdb(self, zip_key, user_id):
    user = CustomUser.objects.get(pk=user_id)

    zip_env = lmdb.open(settings.ZIP_PATH)
    try:
        with zip_env.begin(write=True) as txn:
            zip_data = txn.get(zip_key.encode('utf-8'))
            photo_type = zip_key.split('_')[-1]
            logger.info(f"Processing zip file for key {zip_key}")
            if not zip_data:
                logger.error(f"Error: Zip data not found for key {zip_key}")
                return {'status': 'failure', 'error': 'Zip data not found'}
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as archive:
                if len(archive.namelist()) > settings.LMDB_BATCH_SIZE:
                    return {'status': 'failure', 'error': 'Too many images in zip file'}
                elif len(archive.namelist()) == 0:
                    return {'status': 'failure', 'error': 'No file found'}

                images = [file for file in archive.namelist() if file.endswith(('.png', '.jpg', '.jpeg'))]
                all_img_ids = []
                logger.info(f"Found {len(images)} images in zip file.")
                for file_chunk in chunked_iterable(images, 10):
                    cache = {}
                    for image_name in file_chunk:
                        photo_class = get_photo(photo_type)
                        logger.info(photo_class)
                        logger.info(image_name)
                        if FacePhoto.objects.filter(image_id=image_name).count()>0:
                            logger.info(f"Image {image_name} already exists, skipping...")
                            continue
                        logger.info(f"Processing image {image_name}")
                        photo = photo_class(image_id=image_name, status=AbstractPhoto.Status.UPLOADED)
                        photo.save()

                        # logger.info(photo_class.objects.filter(image_id=image_name).first().to_dict())
                        with archive.open(image_name) as image_file:
                            image_data = image_file.read()
                        cache[image_name] = image_data
                        chain(
                            write_cache_and_process.s(cache,user_id) |
                            process_batch.s(function_type=photo_type).set(queue='image_processing')
                        ).apply_async()

                    all_img_ids.extend(cache.keys())
                    logger.info(f"Processed {len(all_img_ids)} images.")

                user.image_add(images=all_img_ids, photo_type=photo_type)
            txn.delete(zip_key.encode('utf-8'))
            logger.info(f"Zip file processed successfully for key {zip_key}")
        return {'task_id': self.request.id,'status': 'success'}
    except zipfile.BadZipFile:
        logger.error(f"Error: Bad Zip file for key {zip_key}")
        txn.delete(zip_key.encode('utf-8'))
        return {'status': 'failure', 'error': 'Bad Zip file'}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        push_failed_task_id_to_ssd(self.request.id, zip_key=zip_key, user_id=user_id, error=str(e))
        return {'status': 'failure', 'error': str(e)}


def find_similar(faces_embed, photo_embed, k):
    if photo_embed is not None:
        return {'k_faces': pairwise_find(faces_embed, photo_embed, k)}
    else:
        return {'error': 'No face detected in this photo'}


@shared_task(queue='image_processing')
def restart_failed_tasks():
    env = lmdb.open(settings.LMDB_PATH_FTASK, map_size=18000)

    failed_count = 0
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        for task_id, task_info in cursor:
            task_id = task_id.decode('utf-8')
            task_info = json.loads(task_info.decode('utf-8'))
            #TODO: Add more error handling
            error_message = task_info.get('error', '')
            if error_message.startswith('upload'):
                # TODO: process zip file
                pass
            else:
                indices = task_info.get('indices', [])
                process_batch.apply_async([indices, error_message])

            txn.delete(task_id.encode('utf-8'))
            failed_count += 1
        logger.info(f"Restarted {failed_count} failed tasks.")
