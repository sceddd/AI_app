import json
import logging

import requests
from celery import shared_task
from django.shortcuts import get_object_or_404

from AI_Backend import settings
from account.app_models.photos import AbstractPhoto, OCRPhoto

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


@shared_task(bind=True)
def ocr_process(self,indices):
    err = []
    logger.info(f"Processing images {indices}")
    ocr_payload = json.dumps({'idx': indices, 'lmdb_path': settings.LMDB_PATH})
    self.set_state(state='PROGRESS', meta={'image_id': indices})
    try:
        ocr_response = requests.post(settings.TORCHSERVE_URI_OCR, headers={'Content-Type': 'application/json'},
                                     data=ocr_payload)
        ocr_response.raise_for_status()
        ocr_response = ocr_response.json()
        self.update_state(state='OCR_PROGRESS', meta={'ocr_response': ocr_response})
        logger.info(f"OCR response: {ocr_response}")
        [process_ocr_image(ocr_data, err)for ocr_data in ocr_response]
    except requests.exceptions.RequestException as e:
        self.update_state(state='OCR_FAILURE', meta={'error': str(e)})
        err.append(f"Error: {str(e)}")
    return err
