from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings
import logging

from kombu import Queue

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Backend.settings')

app = Celery('AI_Backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)
app.conf.task_queues = {
    Queue('image_upload', routing_key='image.upload'),
    Queue('image_processing', routing_key='image.process')
}

app.conf.task_routes = {
    'your_project.tasks.batch_upload_images_to_mongodb': {'queue': 'image_upload'},
    'your_project.tasks.process_image_data': {'queue': 'image_processing'}
}
logger = logging.getLogger('celery')
handler = logging.FileHandler('./logs/celery.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')