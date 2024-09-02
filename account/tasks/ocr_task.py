import json
import logging

import requests
from celery import shared_task
from django.shortcuts import get_object_or_404
from numpy import error_message

from AI_Backend import settings
from account.app_models.photos import AbstractPhoto, OCRPhoto
from ..project_utils.utils import push_failed_task_id_to_ssd

logger = logging.getLogger(__name__)


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


@shared_task(bind=True, queue='image_processing')
def ocr_process(self, indices):
    err = []
    task_id = self.request.id

    logger.info(f"Processing images {indices}")
    ocr_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})
    try:
        ocr_response = requests.post(settings.TORCHSERVE_URI_OCR, headers={'Content-Type': 'application/json'},
                                     data=ocr_payload)
        ocr_response.raise_for_status()
        ocr_response = ocr_response.json()
        logger.info(f"OCR response: {ocr_response}")

        [process_ocr_image(ocr_data, err) for ocr_data in ocr_response]

    except requests.exceptions.RequestException as e:
        error_message = f'ocr_OCRFailed:{e}'
        push_failed_task_id_to_ssd(task_id, indices=indices, error=error_message)
        logger.error(f"Task {task_id} failed: {error_message}")
        return {'status': 'failure', 'error':  error_message}

    return {'status': 'success', 'error': err}
