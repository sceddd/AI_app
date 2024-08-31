"""
Django settings for myproject project.

Generated by 'django-admin startproject' using Django 5.0.7.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""
import logging.config
import os
import sys
from datetime import timedelta
from pathlib import Path

import redis
from dotenv import load_dotenv
from kombu import Queue
from pymongo import MongoClient

from account.app_models.cluster_models import DimReductionAndClustering

# Build paths inside the project like this: BASE_DIR / 'subdir'.
load_dotenv('.env')


BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'account',
    'rest_framework',
    'rest_framework_simplejwt',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'AI_Backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'AI_Backend.wsgi.application'

AUTH_USER_MODEL = 'account.CustomUser'

DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'account_verification',
        'ENFORCE_SCHEMA': False,
        'CLIENT': {
            'host': os.getenv('MONGO_URI'),
            'port': int(os.getenv('PORT')),
            'username': os.getenv('MONGO_USERNAME'),
            'password': os.getenv('PASSWORD'),
            'authSource': os.getenv('AUTHSOURCE'),
            'authMechanism': os.getenv('AUTHMECHANISM')
        }
    }

}

LOGIN_URL = '/account/login/'

MONGO_CLIENT = MongoClient(
    host=os.getenv('MONGO_URI'),
    port=int(os.getenv('PORT')),
    username=os.getenv('MONGO_USERNAME'),
    password=os.getenv('PASSWORD'),
    authSource=os.getenv('AUTHSOURCE'),
    authMechanism=os.getenv('AUTHMECHANISM')
)

IMAGE_DB = MONGO_CLIENT['face_tables']
# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated', )
}
# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

REDIS_CLIENT = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# celery settings
CELERY_BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
CELERY_RESULT_BACKEND = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

CELERYD_LOG_FILE = './logs/celery.log'
CELERYD_LOG_LEVEL = 'DEBUG'
CELERY_TASK_QUEUES = (
    Queue('image_upload', routing_key='image.upload'),
    Queue('image_processing', routing_key='image.process'),
    Queue('write_cache_and_process', routing_key='image.cache'),
)

CELERY_TASK_ROUTES = {
    'account.tasks.batch_upload_images_to_mongodb': {'queue': 'image_upload'},
    'account.tasks.process_batch': {'queue': 'image_processing'},
    'account.tasks.write_cache_and_process': {'queue': 'write_cache_and_process'}
}

# LMDB settings
LMDB_PATH = os.path.join(BASE_DIR, 'lmdb')
LMDB_BATCH_SIZE = 150
LMDB_PATH_FACE = os.path.join(LMDB_PATH,'face', 'det')
LMDB_LIMIT = 1099511627776
os.makedirs(LMDB_PATH, exist_ok=True)
os.makedirs(LMDB_PATH_FACE, exist_ok=True)

# TorchServe settings
TS_HOST = 'localhost'
TS_PORT = 8080

TORCHSERVE_URI_DET = f'http://{TS_HOST}:{TS_PORT}/predictions/facedet'
TORCHSERVE_URI_REG = f'http://{TS_HOST}:{TS_PORT}/predictions/facereg'

TORCHSERVE_URI_OCR = f'http://{TS_HOST}:{TS_PORT}/predictions/ocr'
TORCHSERVE_URI_OD = f'http://{TS_HOST}:{TS_PORT}/predictions/obdet'


class SpecificTextFilter(logging.Filter):
    def __init__(self, max_length=100):
        super().__init__()
        self.max_length = max_length

    def filter(self, record):

        if "first seen with mtime" in record.msg:
            return False

        if len(record.msg) > self.max_length:

            record.msg = record.msg[:self.max_length] + '...'

        return True


UMAP_DIR = os.path.join(BASE_DIR, 'account', 'umap_weight')
UMAP_WEIGHT_PATH = os.path.join(UMAP_DIR, 'umap_weight.pkl')
if not os.path.exists(UMAP_WEIGHT_PATH):
    print(f"WARNING: UMAP weights not found at {UMAP_WEIGHT_PATH}.\n"
                  f"run python manage.py download_weight command first")

DR_CFG = {
        'param': {
            'n_neighbors': 20,
            'min_dist': 0.2,
            'n_components': 3,
            'metric': 'euclidean'
        },
        'function': 'umap, UMAP'
    }
CL_CFG = {
    'param': {
        'eps': 0.5,
        'min_samples': 4,
        'metric': 'euclidean'
    },
    'function': 'sklearn.cluster, DBSCAN'
}
current_command = sys.argv[1] if len(sys.argv) > 1 else None
CLUSTER_MODEL = None if current_command == 'download_weight' else DimReductionAndClustering(path=UMAP_WEIGHT_PATH,dr_cfg=DR_CFG,cl_cfg=CL_CFG)

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOGGING = {
    "version": 1,  # the dictConfig format version
    "disable_existing_loggers": False,  # retain the default loggers
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "django_file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "./logs/debug.log",
            "filters": ["specific_text_filter"],
        },
        "celery_file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "./logs/celery.log",
            "filters": ["specific_text_filter"],
        },
    },
    "loggers": {
        "django": {
            "handlers": ["django_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "celery":{
            "handlers": ["celery_file"],
            "level": "DEBUG",
            "propagate": True,
        }
    },
    "filters": {
        "specific_text_filter": {
            "()": SpecificTextFilter,
            "max_length": 100,
        },
    },
}

logging.config.dictConfig(LOGGING)

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': False,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUDIENCE': None,
    'ISSUER': None,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'JTI_CLAIM': 'jti',
}