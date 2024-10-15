import json
import logging

from celery import shared_task
from celery.worker.state import requests
from django.shortcuts import get_object_or_404

from AI_Backend import settings
from account.app_models.photos import ObjectDetPhoto, AbstractPhoto
from ..project_utils.utils import push_failed_task_id_to_ssd

logger = logging.getLogger(__name__)


@shared_task(bind=True, queue='image_processing')
def det_process(self, indices,input_words):
    err = []
    task_id = self.request.id
    user_id = indices[0].split('_')[0]
    logger.info(f"Processing images {indices}")
    od_payload = json.dumps({'idx': indices, 'input_txt': input_words,
                             'lmdb_path': settings.LMDB_PATH + f"/user/{user_id}"})

    try:
        det_response = requests.post(settings.TORCHSERVE_URI_OD, headers={'Content-Type': 'application/json'},
                                     data=od_payload)
        det_response.raise_for_status()
        det_response = det_response.json()
        logger.info(f"OD response: {det_response}")

        [process_od_image(od_data, err) for od_data in det_response]

    except requests.exceptions.RequestException as e:
        error_message = f'ob_det_ODFailed:{e}'
        push_failed_task_id_to_ssd(task_id, indices=indices, error=error_message)
        logger.error(f"Task {task_id} failed: {error_message}")
        return {'status': 'failure', 'error':error_message}
    return {'status': 'success', 'error': err}


def process_od_image(od_data, err):
    idx = od_data.get('idx')
    try:
        photo = get_object_or_404(ObjectDetPhoto, image_id=idx)
        logger.info(photo)
        logger.info(od_data)
        # for data in od_data.get('boxes', []):
        #     ob_det = {
        #
        #     }
        #     photo.add_bounding_box(ob_det)
        #     break

        photo.objects_det = od_data.get('objects', []) if od_data.get('objects') else []
        photo.status = AbstractPhoto.Status.RESULT_SAVED
        photo.save()
        logger.info(f"Image {photo} processed successfully.")
    except Exception as e:
        err.append(f"Error processing image {idx}: {e}")
