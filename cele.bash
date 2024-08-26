# Dừng Celery worker
pkill -f 'celery worker'

# Khởi động lại Celery worker
celery -A AI_Backend worker -Q image_upload,write_cache_and_process,image_processing --loglevel=INFO
