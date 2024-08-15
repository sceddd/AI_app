from rest_framework.authentication import get_authorization_header
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.authentication import TokenAuthentication


def is_authenticated(view_func):
    def wrapper(*args, **kwargs):
        request = args[0]
        auth = get_authorization_header(request).split()

        if not auth or auth[0].lower() != b'token':
            raise AuthenticationFailed("No token provided")

        try:
            token = auth[1].decode()
            # Xác thực token bằng TokenAuthentication của DRF
            token_auth = TokenAuthentication()
            user, _ = token_auth.authenticate_credentials(token.encode())
        except Exception as e:
            raise AuthenticationFailed("Invalid token")

        return view_func(user, *args, **kwargs)

    return wrapper