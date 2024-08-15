import functools

from django.conf import settings


def upload_success(func):
    """
    Decorator to set the flag key to 'completed' if the function completes successfully,
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        flag_key = kwargs.get('flag_key', None)
        redis_client = settings.REDIS_CLIENT

        try:
            result = func(*args, **kwargs)
            if flag_key:
                redis_client.set(flag_key, 'completed')
            return result
        except Exception as e:
            if flag_key:
                redis_client.set(flag_key, 'failed')

            raise e

    return wrapper
