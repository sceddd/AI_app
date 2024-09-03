import json

import lmdb
from celery import shared_task
from django.conf import settings

from .account_utils import create_module

env = lmdb.open(settings.LMDB_PATH_FTASK, map_size=settings.LMDB_LIMIT)
r = settings.REDIS_CLIENT
publisher = r.pubsub()


def push_failed_task_id_to_ssd(task_id, **kwargs):
    with env.begin(write=True) as txn:
        data = json.dumps(kwargs)
        txn.put(task_id.encode('utf-8'), data.encode('utf-8'))


def get_data_redis(id):
    value = r.get(id)
    return value if value is not None else None


def publish_new_results(idx, **kwargs):
    results_idx = 'res_{}'.format(idx)
    data = json.dumps(kwargs)
    r.publish('new_results', json.dumps({'id': results_idx, 'data': data}))


@shared_task(queue='image_processing')
def listen_to_redis(module_str, **kwargs):
    pubsub = r.pubsub()
    pubsub.subscribe('new_results')

    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            result_id = data.get('id')
            result_data = json.loads(data.get('data', '{}'))

            callback_func = create_module(module_str)
            if callback_func:
                callback_func.delay(result_id, result_data,**kwargs)
            else:
                raise ValueError(f"Callback function not found in {module_str}")