from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings
import logging

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Backend.settings')

app = Celery('AI_Backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

# Set Celery task queues and routes from Django settings
app.conf.task_queues = settings.CELERY_TASK_QUEUES
app.conf.task_routes = settings.CELERY_TASK_ROUTES

# Ensure Celery uses the logging configuration from Django settings
# No need to define the logger manually here as Celery will use Django's logging configuration


@app.task(bind=True)
def debug_task(self):
    logger = logging.getLogger("celery")
    logger.debug(f'Request: {self.request!r}')
