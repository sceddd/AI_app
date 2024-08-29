import json
import logging

from celery import shared_task
from celery.worker.state import requests
from django.shortcuts import get_object_or_404

from AI_Backend import settings
from account.app_models.photos import ObjectDetPhoto, AbstractPhoto

logger = logging.getLogger(__name__)


@shared_task
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
        photo.bounding_boxes = od_data.get('boxes', []) if od_data.get('boxes') else []
        photo.objects_det = od_data.get('objects', []) if od_data.get('objects') else []
        photo.status = AbstractPhoto.Status.RESULT_SAVED
        photo.save()
        logger.info(f"Image {photo} processed successfully.")
    except Exception as e:
        err.append(f"Error processing image {idx}: {e}")