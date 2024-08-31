from django.urls import path
from .views import *

urlpatterns = [
    path('register/', register_user, name='register'),
    path('task_status/<str:task_id>/', task_status, name='task_status'),
    path('login/', login_user, name='login'),
    path('upload/', upload, name='upload'),
    path('get_image/<str:image_id>/', get_image_from_gridfs, name='get_image'),
    path('user_photos/', user_photos, name='user_photos'),
    path('logout/', logout_user, name='logout'),
    path('s_face/', get_similar_faces, name='get_similar_faces'),
]


