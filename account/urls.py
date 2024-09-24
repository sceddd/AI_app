from django.urls import path
from .views import *

urlpatterns = [
    path('register/', register_user, name='register'),
    path('login/', login_user, name='login'),
    path('upload/', upload, name='upload'),
    path('get_image/', get_image, name='get_image'),
    path('user_photos/', user_photos, name='user_photos'),
    path('logout/', logout_user, name='logout'),
    path('s_face/', get_similar_faces, name='get_similar_faces'),
    path('task_status/<str:task_id>/', check_task_status, name='check_task_status'),
    path('token/refresh/', refresh_access_token, name='refresh_token'),
]
