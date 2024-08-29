# Dừng Celery worker
pkill -f 'celery worker'

# Khởi động lại Celery worker
FLOWER_UNAUTHENTICATED_API=true celery -A AI_Backend worker -Q image_upload,write_cache_and_process,image_processing -E --loglevel=INFO
