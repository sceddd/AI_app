# Dừng Celery worker
pkill -f 'celery worker'

# Khởi động lại Celery worker
celery -A AI_Backend worker --loglevel=INFO
