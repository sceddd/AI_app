import json

import lmdb
from django.conf import settings
from django.http import JsonResponse

from ..app_models.photos import get_photo_class

env = lmdb.open(settings.LMDB_PATH_FTASK, map_size=settings.LMDB_LIMIT)
result_env = lmdb.open(settings.LMDB_PATH_RESULT, map_size=settings.LMDB_LIMIT)
r = settings.REDIS_CLIENT
publisher = r.pubsub()


def push_failed_task_id_to_ssd(task_id, **kwargs):
    with env.begin(write=True) as txn:
        data = json.dumps(kwargs)
        txn.put(task_id.encode('utf-8'), data.encode('utf-8'))


def publish_new_results(idx, **kwargs):
    results_idx = 'res_{}'.format(idx)
    with result_env.begin(write=True) as txn:
        data = json.dumps(kwargs)
        txn.put(results_idx.encode('utf-8'), data.encode('utf-8'))


def get_data(idx):
    with result_env.begin() as txn:
        data = txn.get(idx.encode('utf-8'))
        return json.loads(data.decode('utf-8'))


def get_photo(photo_type):
    if photo_type not in ['face', 'ocr', 'ob_det']:
        return JsonResponse({'error': 'Invalid photo type'}, status=400)
    return get_photo_class(photo_type)