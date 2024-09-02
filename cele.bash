# Dừng Celery worker
pkill -f 'celery worker'

celery -A AI_Backend worker -Q image_upload,write_cache_and_process,image_processing,celery -E --loglevel=INFO
