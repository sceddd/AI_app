import json
import logging

from django.shortcuts import get_object_or_404
import requests

from AI_Backend import settings
from account.app_models.photos import ObjectDetPhoto, AbstractPhoto, BoundingBox


logger = logging.getLogger(__name__)


def det_process(indices,input_words):
    err = []
    user_id = indices[0].split('_')[0]
    input_words = input_words.split(',')

    logger.info(f"Processing images {indices}")
    logger.info(f"input_words: {input_words}")
    od_payload = json.dumps({'idx': [indices], 'input_txt': input_words,
                             'lmdb_path': settings.LMDB_PATH + f"/user/{user_id}"})

    try:
        det_response = requests.post(settings.TORCHSERVE_URI_OD, headers={'Content-Type': 'application/json'},
                                     data=od_payload)
        det_response.raise_for_status()
        det_response = det_response.json()
        logger.info(f"OD response: {det_response}")
        idx = det_response.get('idx')
        photo = get_object_or_404(ObjectDetPhoto, image_id=idx)

        process_od_image(idx,photo,det_response, err)

    except requests.exceptions.RequestException as e:
        error_message = f'ob_det_ODFailed:{e}'
        return {'status': 'failure', 'error':error_message}
    return photo.to_dict()


def process_od_image(idx, photo, od_data, err):
    logger.info(f"Processing OD image: {idx}")
    try:

        objects = od_data['objects'][0]
        logger.info(objects)
        logger.info(f"Deleting existing bounding boxes for image {idx}")
        if hasattr(photo, 'bounding_boxes') and isinstance(photo.bounding_boxes, list):
            photo.bounding_boxes = []  # Clear the list
            logger.info("Existing bounding boxes cleared.")
        else:
            logger.warning("photo.bounding_boxes is not a list. Skipping deletion.")

        logger.info(f"Detected {len(objects.get('boxes', []))} objects")

        for idx in range(len(objects['boxes'])):
            data = {
                'bboxes': objects['boxes'][idx],
                'conf': objects['conf'][idx],
                'cls': objects['classes'][idx]
            }

            photo.add_bounding_box(BoundingBox(**data))
        photo.status = AbstractPhoto.Status.RESULT_SAVED
        photo.save()

    except Exception as e:
        err.append(f"Error processing image {idx}: {e}")
