import json
import logging

import requests
from celery import shared_task
from django.shortcuts import get_object_or_404

from AI_Backend import settings
from account.app_models.photos import AbstractPhoto, OCRPhoto, BoundingBox
from ..project_utils.utils import push_failed_task_id_to_ssd

logger = logging.getLogger(__name__)


def process_ocr_image(ocr_data, err):
    idx = ocr_data.get('idx')
    try:
        photo = get_object_or_404(OCRPhoto, image_id=idx)
        print(ocr_data)
        for idx in range(len(ocr_data['bboxes'])):
            print(ocr_data['bboxes'][idx])
            print(ocr_data['cls'][idx])
            data = {
                'bboxes': ocr_data['bboxes'][idx],
                'conf': ocr_data.get('cnf', 0.0),
                'cls': ocr_data['cls'][idx]
            }
            photo.add_bounding_box(BoundingBox(**data))
        photo.status = AbstractPhoto.Status.RESULT_SAVED
        photo.save()
        logger.info(f"Image {photo} processed successfully.")

    except Exception as e:
        err.append(f"Error processing image {idx}: {e}")


@shared_task(bind=True, queue='image_processing')
def ocr_process(self, indices):
    err = []
    task_id = self.request.id
    user_id = indices[0].split('_')[0]
    logger.info(f"Processing images {indices}")
    ocr_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH + f"/user/{user_id}"})
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
